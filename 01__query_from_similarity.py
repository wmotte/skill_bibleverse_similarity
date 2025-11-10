#!/usr/bin/env python3
#
# Wim Otte (w.m.otte@umcutrecht.nl)
"""Query a pre-built verse similarity bundle."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return the top-N most similar verses from a cached bundle."
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "verse_similarity_bundle_int8.joblib",
        help="Path to the serialized embeddings bundle produced by bible_similarity.py.",
    )
    parser.add_argument(
        "--reference",
        help="Reference like 'John 1 1' or 'Genesis 1:1'; positional tokens also work.",
    )
    parser.add_argument("--book", help="Book name when --reference is not provided.")
    parser.add_argument("--chapter", type=int, help="Chapter number.")
    parser.add_argument("--verse", type=int, help="Verse number.")
    parser.add_argument("--top", type=int, default=5, help="Number of matches to return.")
    parser.add_argument(
        "positional",
        nargs="*",
        help="Optional positional reference tokens such as `Ruth 1 2`.",
    )
    return parser.parse_args()


NUM_PATTERN = re.compile(r"\d+")
BOOK_NORMALIZER = re.compile(r"[^a-z0-9]+")


def parse_numeric(value: str, field_name: str) -> int:
    match = NUM_PATTERN.search(value or "")
    if not match:
        raise SystemExit(f"Unable to parse {field_name} value from '{value}'.")
    return int(match.group())


def make_key(book: str, chapter: int, verse: int) -> str:
    return f"{book.strip()}|{chapter}|{verse}"


def parse_reference_string(reference: str) -> Tuple[str, int, int]:
    if not reference:
        raise SystemExit("Empty reference string.")

    normalized = reference.strip()
    for sep in (":", ".", ","):
        normalized = normalized.replace(sep, " ")
    tokens = normalized.split()
    if len(tokens) < 3:
        raise SystemExit(
            "Reference must include a book name plus chapter and verse numbers "
            "(e.g. 'Ruth 1 2' or 'Ruth 1:2')."
        )

    chapter = parse_numeric(tokens[-2], "chapter")
    verse = parse_numeric(tokens[-1], "verse")
    book = " ".join(tokens[:-2]).strip()
    if not book:
        raise SystemExit("Reference book name is missing.")
    return book, chapter, verse


def resolve_reference(args: argparse.Namespace) -> Tuple[str, int, int]:
    if args.reference:
        return parse_reference_string(args.reference)
    if args.book and args.chapter is not None and args.verse is not None:
        return args.book.strip(), args.chapter, args.verse
    if args.positional:
        positional_ref = " ".join(args.positional).strip()
        if positional_ref:
            return parse_reference_string(positional_ref)
    raise SystemExit(
        "Provide a reference via --reference, --book/--chapter/--verse, or positional tokens."
    )


def normalize_book_name(book: str) -> str:
    return BOOK_NORMALIZER.sub("", (book or "").lower())


def build_book_alias_map(verses: List[Dict[str, object]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for verse in verses:
        canonical = verse["book"]
        abbr = verse.get("abbr", "")
        for candidate in (canonical, abbr):
            normalized = normalize_book_name(candidate)
            if normalized and normalized not in alias_map:
                alias_map[normalized] = canonical
    return alias_map


def canonicalize_book_name(book: str, alias_map: Dict[str, str]) -> str:
    normalized = normalize_book_name(book)
    if normalized in alias_map:
        return alias_map[normalized]
    suggestions = get_close_matches(normalized, alias_map.keys(), n=3, cutoff=0.6)
    if suggestions:
        candidates = sorted({alias_map[key] for key in suggestions})
        if len(candidates) == 1:
            return candidates[0]
        raise SystemExit(f"Ambiguous book name '{book}'. Did you mean: {', '.join(candidates)}?")
    raise SystemExit(f"Unknown book name '{book}'. Please provide a valid full name or abbreviation.")


def restore_svd_int8_matrix(
    encoded: np.ndarray,
    payload: Dict[str, object],
) -> np.ndarray:
    projection = payload.get("projection")
    scales = payload.get("svd_scales")
    if projection is None or scales is None:
        raise SystemExit(
            "Bundle stores matrix_format=svd_int8 but is missing "
            "the required 'projection' or 'svd_scales' entries."
        )
    projection_arr = np.asarray(projection, dtype=np.float32)
    scales_arr = np.asarray(scales, dtype=np.float32)
    if encoded.ndim != 2:
        raise SystemExit("Compressed matrix must be 2D.")
    if projection_arr.shape[0] != encoded.shape[1]:
        raise SystemExit(
            "Projection row count does not match compressed matrix width. "
            "Rebuild the bundle."
        )
    if scales_arr.shape[0] != encoded.shape[1]:
        raise SystemExit(
            "svd_scales length does not match compressed matrix width. "
            "Rebuild the bundle."
        )
    reduced = encoded.astype(np.float32) * scales_arr
    return reduced @ projection_arr


def load_bundle(bundle_path: Path) -> Tuple[np.ndarray, List[str], List[Dict[str, object]]]:
    if not bundle_path.exists():
        raise SystemExit(f"Bundle file not found: {bundle_path}")

    payload = joblib.load(bundle_path)
    try:
        matrix = payload["matrix"]
        keys = payload["keys"]
        verses = payload["verses"]
    except KeyError as exc:
        raise SystemExit(
            f"Bundle {bundle_path} does not contain the expected data. Rebuild the bundle."
        ) from exc

    if not isinstance(matrix, np.ndarray):
        raise SystemExit(
            f"Bundle {bundle_path} stores an unsupported matrix type. "
            "Regenerate it with bible_similarity.py to obtain dense embeddings."
        )
    meta = payload.get("meta") or {}
    matrix_format = meta.get("matrix_format")
    if matrix_format == "svd_int8":
        matrix = restore_svd_int8_matrix(matrix, payload)

    quant_meta = meta.get("quantization") if isinstance(meta, dict) else None

    if matrix.dtype == np.float32:
        pass
    elif matrix.dtype == np.float16:
        matrix = matrix.astype(np.float32)
    elif matrix.dtype == np.int8:
        if not quant_meta or quant_meta.get("mode") != "symmetric_int8":
            raise SystemExit(
                f"Bundle {bundle_path} stores int8 data without quantization metadata."
            )
        scale = float(quant_meta.get("scale") or 0.0)
        if scale <= 0:
            raise SystemExit(f"Bundle {bundle_path} has invalid quantization scale {scale}.")
        matrix = matrix.astype(np.float32) / scale
    else:
        raise SystemExit(
            f"Bundle {bundle_path} uses unsupported dtype {matrix.dtype}. "
            "Regenerate or convert it to float32/float16/int8."
        )
    if matrix.shape[0] != len(keys) or len(keys) != len(verses):
        raise SystemExit(f"Bundle {bundle_path} appears corrupt. Rebuild the bundle.")
    return matrix, keys, verses


def lookup_query_index(
    verse_lookup: Dict[str, Dict[str, object]],
    key_to_index: Dict[str, int],
    ref_key: str,
) -> int:
    if ref_key not in verse_lookup:
        raise SystemExit(f"Verse not found in bundle metadata: {ref_key}")
    try:
        return key_to_index[ref_key]
    except KeyError as exc:
        raise SystemExit(
            "Verse missing from the embedding matrix. Regenerate the bundle if necessary."
        ) from exc


def collect_similar_verses(
    matrix: np.ndarray,
    keys: List[str],
    verse_lookup: Dict[str, Dict[str, object]],
    query_index: int,
    top_n: int,
) -> List[Dict[str, object]]:
    if top_n < 1:
        raise SystemExit("--top must be a positive integer.")
    query_vector = matrix[query_index]
    similarities = np.dot(matrix, query_vector)
    similarities[query_index] = -1.0

    candidate_count = min(len(similarities), max(top_n * 4, top_n))
    candidate_idx = np.argpartition(similarities, -candidate_count)[-candidate_count:]
    sorted_idx = candidate_idx[np.argsort(similarities[candidate_idx])[::-1]]

    results: List[Dict[str, object]] = []
    seen_keys = set()
    for idx in sorted_idx:
        key = keys[idx]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        verse = verse_lookup.get(key)
        if not verse:
            continue
        results.append(
            {
                "book": verse["book"],
                "chapter": verse["chapter"],
                "verse": verse["verse"],
                "text": verse["text"],
                "id": verse["id"],
                "similarity": round(float(similarities[idx]), 6),
            }
        )
        if len(results) >= top_n:
            break
    return results


def main() -> None:
    args = parse_args()
    requested_book, chapter, verse_number = resolve_reference(args)
    matrix, keys, verses = load_bundle(args.bundle_path)

    alias_map = build_book_alias_map(verses)
    book = canonicalize_book_name(requested_book, alias_map)
    target_key = make_key(book, chapter, verse_number)

    verse_lookup = {item["key"]: item for item in verses}
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    query_idx = lookup_query_index(verse_lookup, key_to_index, target_key)

    matches = collect_similar_verses(matrix, keys, verse_lookup, query_idx, args.top)
    payload = {
        "query": {
            "book": book,
            "chapter": chapter,
            "verse": verse_number,
            "key": target_key,
            "text": verse_lookup[target_key]["text"],
        },
        "matches": matches,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
