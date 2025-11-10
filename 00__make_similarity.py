#!/usr/bin/env python3
#
# Wim Otte (w.m.otte@umcutrecht.nl)
"""Compute and query Bible verse similarities using cached semantic embeddings."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return the top-N most similar verses for a given reference."
    )
    parser.add_argument(
        "--reference",
        help="Bible reference such as 'Ruth 1.2' or 'Genesis 1:1'.",
    )
    parser.add_argument("--book", help="Book name when --reference is not provided.")
    parser.add_argument("--chapter", type=int, help="Chapter number.")
    parser.add_argument("--verse", type=int, help="Verse number.")
    parser.add_argument("--top", type=int, default=5, help="Number of matches to return.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).parent / "input" / "lxx_and_nt.tsv",
        help="Path to the TSV file containing the LXX+NT text.",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "verse_similarity_bundle.joblib",
        help="Path of the serialized embeddings bundle.",
    )
    parser.add_argument(
        "--embedding-model",
        default="bge-m3:latest",
        help="Ollama embedding model tag to use (default: bge-m3:latest).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel embedding requests to issue when rebuilding.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=60,
        help="Per-request timeout (seconds) for Ollama embedding calls.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the embeddings cache before serving the query.",
    )
    parser.add_argument(
        "positional",
        nargs="*",
        help="Optional reference like 'Ruth 1 2' when flags are omitted.",
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
        raise ValueError("Empty reference string.")

    normalized = reference.strip()
    for sep in (":", ".", ","):
        normalized = normalized.replace(sep, " ")
    tokens = normalized.split()
    if len(tokens) < 3:
        raise ValueError(
            "Reference must include a book name plus chapter and verse numbers "
            "(e.g. 'Ruth 1 2' or 'Ruth 1:2')."
        )

    chapter = parse_numeric(tokens[-2], "chapter")
    verse = parse_numeric(tokens[-1], "verse")
    book = " ".join(tokens[:-2]).strip()
    if not book:
        raise ValueError("Reference book name is missing.")

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
        raise SystemExit(
            f"Ambiguous book name '{book}'. Did you mean: {', '.join(candidates)}?"
        )
    raise SystemExit(
        f"Unknown book name '{book}'. Please provide a valid full name or abbreviation."
    )


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def build_ollama_endpoint(base_url: str) -> str:
    base = (base_url or "").strip()
    if not base:
        base = "http://localhost:11434"
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    return base.rstrip("/") + "/api/embeddings"


def request_embedding(
    text: str,
    model: str,
    endpoint: str,
    timeout: int,
) -> np.ndarray:
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Failed to contact Ollama at {endpoint}. "
            "Ensure `ollama serve` is running and accessible."
        ) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid response from Ollama: {body}") from exc
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise SystemExit(f"Ollama response missing embedding vector: {data}")
    return np.asarray(embedding, dtype=np.float32)


def generate_semantic_embeddings(
    texts: List[str],
    model: str,
    endpoint: str,
    concurrency: int,
    timeout: int,
) -> np.ndarray:
    total = len(texts)
    if total == 0:
        return np.zeros((0, 0), dtype=np.float32)
    workers = max(1, concurrency)
    results: List[np.ndarray] = [None] * total  # type: ignore
    counter = 0
    counter_lock = threading.Lock()

    def embed_single(task: Tuple[int, str]) -> Tuple[int, np.ndarray]:
        index, verse_text = task
        return index, request_embedding(verse_text, model, endpoint, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, vector in executor.map(embed_single, enumerate(texts)):
            results[idx] = vector
            with counter_lock:
                counter += 1
                if counter % 250 == 0 or counter == total:
                    print(
                        f"Embedded {counter}/{total} verses...",
                        file=sys.stderr,
                        flush=True,
                    )

    stacked = np.vstack(results).astype(np.float32)  # type: ignore[arg-type]
    return normalize_vectors(stacked)


def load_verses(tsv_path: Path) -> List[Dict[str, object]]:
    if not tsv_path.exists():
        raise SystemExit(f"Input file not found: {tsv_path}")

    verses: List[Dict[str, object]] = []
    with tsv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header or len(header) < 5:
            raise SystemExit("TSV header is missing required columns.")

        current_book = None
        inferred_chapter = 0

        for row in reader:
            if not row:
                continue
            book = row[0].strip()
            abb = row[1].strip()
            try:
                book_id = int(row[2])
            except ValueError:
                book_id = row[2].strip()

            if book != current_book:
                current_book = book
                inferred_chapter = 0

            if len(row) == 6:
                chapter = parse_numeric(row[3], "chapter")
                verse_no = parse_numeric(row[4], "verse")
                text = row[5].strip()
                inferred_chapter = chapter
            elif len(row) == 5:
                verse_no = parse_numeric(row[3], "verse")
                text = row[4].strip()
                if verse_no == 1:
                    inferred_chapter += 1
                if inferred_chapter == 0:
                    inferred_chapter = 1
                chapter = inferred_chapter
            else:
                raise SystemExit(f"Unexpected column count ({len(row)}) in TSV line: {row}")

            verse = {
                "book": book,
                "abbr": abb,
                "id": book_id,
                "chapter": chapter,
                "verse": verse_no,
                "text": text,
            }
            verse["key"] = make_key(verse["book"], verse["chapter"], verse["verse"])
            verses.append(verse)

    if not verses:
        raise SystemExit("No verses found in the TSV file.")
    return verses


def build_embeddings(
    verses: List[Dict[str, object]],
    embedding_model: str,
    ollama_url: str,
    concurrency: int,
    timeout: int,
) -> np.ndarray:
    texts = [item["text"] for item in verses]
    endpoint = build_ollama_endpoint(ollama_url)
    print(
        f"Generating embeddings with {embedding_model} via {endpoint}... ",
        file=sys.stderr,
    )
    return generate_semantic_embeddings(texts, embedding_model, endpoint, concurrency, timeout)


def restore_svd_int8_matrix(
    encoded: np.ndarray,
    payload: Dict[str, Any],
) -> np.ndarray:
    projection = payload.get("projection")
    scales = payload.get("svd_scales")
    if projection is None or scales is None:
        raise SystemExit(
            "Bundle stores matrix_format=svd_int8 but is missing projection/scales. "
            "Rebuild the embeddings cache."
        )
    projection_arr = np.asarray(projection, dtype=np.float32)
    scales_arr = np.asarray(scales, dtype=np.float32)
    if encoded.ndim != 2:
        raise SystemExit("Compressed matrix must be two-dimensional.")
    if projection_arr.shape[0] != encoded.shape[1] or scales_arr.shape[0] != encoded.shape[1]:
        raise SystemExit("SVD metadata mismatch detected. Rebuild the embeddings cache.")
    reduced = encoded.astype(np.float32) * scales_arr
    return reduced @ projection_arr


def load_bundle(bundle_path: Path) -> Tuple[np.ndarray, List[str], List[Dict[str, object]], Dict[str, object]]:
    payload = joblib.load(bundle_path)
    try:
        matrix = payload["matrix"]
        keys = payload["keys"]
        verses = payload["verses"]
        meta = payload.get("meta", {})
    except KeyError as exc:
        raise SystemExit(
            f"Bundle file {bundle_path} is missing required data. Rebuild with --rebuild."
        ) from exc
    matrix_format = meta.get("matrix_format")
    if matrix_format == "svd_int8":
        matrix = restore_svd_int8_matrix(matrix, payload)
    if matrix.shape[0] != len(keys) or len(keys) != len(verses):
        raise SystemExit(f"Bundle file {bundle_path} is inconsistent. Rebuild with --rebuild.")
    if not isinstance(matrix, np.ndarray):
        raise SystemExit(
            "Legacy embedding cache detected. Please rebuild with --rebuild to switch to semantic embeddings."
        )
    return matrix, keys, verses, meta


def save_bundle(
    bundle_path: Path,
    matrix: np.ndarray,
    verses: List[Dict[str, object]],
    embedding_model: str,
) -> Tuple[np.ndarray, List[str], List[Dict[str, object]], Dict[str, object]]:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix": matrix,
        "keys": [item["key"] for item in verses],
        "verses": verses,
        "meta": {"embedding_model": embedding_model},
    }
    joblib.dump(payload, bundle_path, compress=3)
    return matrix, payload["keys"], verses, payload["meta"]


def ensure_embeddings(
    data_path: Path,
    bundle_path: Path,
    rebuild: bool,
    embedding_model: str,
    ollama_url: str,
    concurrency: int,
    timeout: int,
) -> Tuple[np.ndarray, List[str], List[Dict[str, object]]]:
    if bundle_path.exists() and not rebuild:
        matrix, keys, verses, meta = load_bundle(bundle_path)
        cached_model = meta.get("embedding_model")
        if cached_model and cached_model != embedding_model:
            raise SystemExit(
                f"Existing bundle was built with model '{cached_model}'. "
                "Re-run with --rebuild to regenerate using the requested model."
            )
        return matrix, keys, verses

    verses = load_verses(data_path)
    matrix = build_embeddings(verses, embedding_model, ollama_url, concurrency, timeout)
    matrix, keys, verses, _ = save_bundle(bundle_path, matrix, verses, embedding_model)
    return matrix, keys, verses


def lookup_query_index(
    verse_lookup: Dict[str, Dict[str, str]],
    key_to_index: Dict[str, int],
    ref_key: str,
) -> int:
    if ref_key not in verse_lookup:
        raise SystemExit(f"Verse not found in TSV data: {ref_key}")
    try:
        return key_to_index[ref_key]
    except KeyError as exc:
        raise SystemExit("Verse not present in the cached embeddings. Rebuild cache.") from exc


def collect_similar_verses(
    matrix: np.ndarray,
    keys: List[str],
    verse_lookup: Dict[str, Dict[str, str]],
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

    matrix, keys, verses = ensure_embeddings(
        args.data_path,
        args.bundle_path,
        args.rebuild,
        args.embedding_model,
        args.ollama_url,
        args.concurrency,
        args.ollama_timeout,
    )
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
