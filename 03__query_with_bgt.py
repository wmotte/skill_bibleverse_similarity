#!/usr/bin/env python3
"""Query the similarity bundle and attach Bijbel in Gewone Taal verses."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from difflib import get_close_matches

NUM_PATTERN = re.compile(r"\d+")
BOOK_NORMALIZER = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return verse similarities and nearby BGT verses."
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=Path(__file__).parent / "misc" / "verse_similarity_bundle_int8.joblib",
        help="Path to the serialized embeddings bundle.",
    )
    parser.add_argument(
        "--bgt-path",
        type=Path,
        default=Path(__file__).parent / "misc" / "bgt_with_titles.joblib",
        help="Path to the Bijbel in Gewone Taal TSV export or its binary dump.",
    )
    parser.add_argument(
        "--lxx-mapping",
        type=Path,
        default=Path(__file__).parent / "misc" / "lxx_to_hebrew_mapping.joblib",
        help="Path to the JSON/Joblib file that maps LXX references to Hebrew numbering.",
    )
    parser.add_argument(
        "--reference",
        help="Reference like 'John 3 16' or 'Genesis 1:1'; positional tokens also work.",
    )
    parser.add_argument("--book", help="Book name when --reference is not provided.")
    parser.add_argument("--chapter", type=int, help="Chapter number.")
    parser.add_argument("--verse", type=int, help="Verse number.")
    parser.add_argument("--top", type=int, default=10, help="Number of matches to return.")
    parser.add_argument(
        "positional",
        nargs="*",
        help="Optional positional reference tokens such as `Ruth 1 2`.",
    )
    return parser.parse_args()


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
    raise SystemExit(f"Unknown book name '{book}'. Please provide a valid name or abbreviation.")


def restore_svd_int8_matrix(
    encoded: np.ndarray,
    payload: Dict[str, Any],
) -> np.ndarray:
    projection = payload.get("projection")
    scales = payload.get("svd_scales")
    if projection is None or scales is None:
        raise SystemExit(
            "Bundle stores matrix_format=svd_int8 but lacks 'projection' or 'svd_scales'. "
            "Regenerate it with 02__compress_bundle.py."
        )
    projection_arr = np.asarray(projection, dtype=np.float32)
    scales_arr = np.asarray(scales, dtype=np.float32)
    if encoded.ndim != 2:
        raise SystemExit("Compressed matrix must be 2-dimensional.")
    if projection_arr.shape[0] != encoded.shape[1] or scales_arr.shape[0] != encoded.shape[1]:
        raise SystemExit("SVD metadata mismatch detected. Rebuild the bundle.")
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
                "key": verse["key"],
                "similarity": round(float(similarities[idx]), 6),
            }
        )
        if len(results) >= top_n:
            break
    return results


def sanitize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def format_bgt_summary(bgt_info: Dict[str, Any]) -> str:
    if not bgt_info:
        return "BGT data unavailable"
    if not bgt_info.get("available"):
        return sanitize_text(bgt_info.get("reason") or "BGT data unavailable")
    references = bgt_info.get("references") or []
    parts: List[str] = []
    for ref in references:
        book = sanitize_text(ref.get("book") or "")
        chapter = ref.get("chapter")
        verses = ref.get("verses") or []
        verse_segments = []
        for verse in verses:
            label = verse.get("raw_verse") or f"{verse.get('start')}-{verse.get('end')}"
            verse_text = sanitize_text(verse.get("text") or "")
            segment = f"{book} {chapter}:{label} {verse_text}".strip()
            verse_segments.append(segment)
        if verse_segments:
            parts.append(" | ".join(verse_segments))
    return " || ".join(parts) if parts else "BGT data unavailable"


@dataclass
class BgtEntry:
    book_key: str
    display_book: str
    chapter: int
    raw_verse: str
    start: int
    end: int
    text: str


@dataclass
class HebrewRef:
    book: str
    chapter: int
    verse_start: int
    verse_end: int


def load_lxx_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"LXX to Hebrew mapping file not found: {path}")
    if path.suffix.lower() == ".joblib":
        raw = joblib.load(path)
    else:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)

    normalized = {normalize_book_name(name): info for name, info in raw.items()}
    alias_pairs = {
        "1samuel": "1kingdomslxx",
        "2samuel": "2kingdomslxx",
        "1kings": "3kingdomslxx",
        "2kings": "4kingdomslxx",
        "wisdomofsolomon": "wisdom",
        "songofsolomon": "songofsongs",
        "songofsongs": "songofsongs",
        "sirach": "sirach",
        "belandthedragon": "belandthedragon",
    }
    for alias, source in alias_pairs.items():
        info = normalized.get(source)
        if info is not None:
            normalized[alias] = info
    return normalized


def parse_verse_range(value: str) -> Tuple[int, int]:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Empty verse label.")
    if "-" in cleaned:
        start_str, end_str = cleaned.split("-", 1)
        return int(start_str), int(end_str)
    number = int(cleaned)
    return number, number


def load_bgt_verses(path: Path) -> Dict[str, Dict[int, List[BgtEntry]]]:
    if not path.exists():
        raise SystemExit(f"BGT file not found: {path}")

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        with gzip.open(path, "rb") as handle:
            raw_bytes = handle.read()
    else:
        raw_bytes = path.read_bytes()

    payload: Optional[Dict[str, Any]]
    try:
        payload = joblib.load(io.BytesIO(raw_bytes))
        if not isinstance(payload, dict):
            payload = None
    except Exception:
        payload = None

    if payload is not None:
        index: Dict[str, Dict[int, List[BgtEntry]]] = {}

        def coerce_entry(raw_entry: object, normalized_key: str) -> BgtEntry:
            if isinstance(raw_entry, BgtEntry):
                return BgtEntry(
                    book_key=normalized_key,
                    display_book=raw_entry.display_book,
                    chapter=int(raw_entry.chapter),
                    raw_verse=str(raw_entry.raw_verse),
                    start=int(raw_entry.start),
                    end=int(raw_entry.end),
                    text=raw_entry.text,
                )
            data = dict(raw_entry)
            return BgtEntry(
                book_key=normalized_key,
                display_book=data.get("display_book") or data.get("book") or "",
                chapter=int(data.get("chapter", 0)),
                raw_verse=str(data.get("raw_verse") or data.get("verse") or ""),
                start=int(data.get("start", 0)),
                end=int(data.get("end", 0)),
                text=data.get("text") or "",
            )

        for book_key, chapters in payload.items():
            normalized_key = normalize_book_name(book_key)
            chapter_map: Dict[int, List[BgtEntry]] = {}
            for chapter_raw, entries in chapters.items():
                chapter_num = int(chapter_raw)
                bucket = [coerce_entry(entry, normalized_key) for entry in entries]
                chapter_map[chapter_num] = bucket
            index[normalized_key] = chapter_map
        return index

    index: Dict[str, Dict[int, List[BgtEntry]]] = {}
    text_buffer = io.StringIO(raw_bytes.decode("utf-8"))
    reader = csv.DictReader(text_buffer, delimiter="\t")
    for row in reader:
        if (row.get("type") or "").strip() != "verse":
            continue
        book = (row.get("book") or "").strip()
        chapter_raw = (row.get("chapter") or "").strip()
        verse_raw = (row.get("verse") or "").strip()
        text = (row.get("text") or "").strip()
        if not (book and chapter_raw and verse_raw and text):
            continue
        try:
            chapter = int(chapter_raw)
            start, end = parse_verse_range(verse_raw)
        except ValueError:
            continue
        book_key = normalize_book_name(book)
        entries = index.setdefault(book_key, {}).setdefault(chapter, [])
        entries.append(
            BgtEntry(
                book_key=book_key,
                display_book=book,
                chapter=chapter,
                raw_verse=verse_raw,
                start=start,
                end=end,
                text=text,
            )
        )
    return index


PSALM_BOUNDARIES = {
    (9, 10): 21,
    (114, 115): 8,
}


def map_psalm_reference(chapter: int, verse: int, info: Dict[str, Any]) -> List[HebrewRef]:
    mapping = info.get("detailed_mapping") or {}
    entry = mapping.get(str(chapter))
    if not entry:
        return [HebrewRef("Psalms", chapter, verse, verse)]

    target = entry.get("hebrew_psalm")
    if isinstance(target, int):
        return [HebrewRef("Psalms", int(target), verse, verse)]

    target_str = str(target)
    if ":" in target_str:
        psalm_str, verse_part = target_str.split(":", 1)
        psalm_num = int(psalm_str)
        if "-" in verse_part:
            start_str, end_str = verse_part.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = end = int(verse_part)
        mapped_verse = start + max(0, verse - 1)
        mapped_verse = max(start, min(end, mapped_verse))
        return [HebrewRef("Psalms", psalm_num, mapped_verse, mapped_verse)]

    if "-" in target_str:
        first_str, second_str = target_str.split("-", 1)
        first_psalm, second_psalm = int(first_str), int(second_str)
        boundary = PSALM_BOUNDARIES.get((first_psalm, second_psalm))
        if boundary:
            if verse <= boundary:
                return [HebrewRef("Psalms", first_psalm, verse, verse)]
            adjusted = max(1, verse - boundary)
            return [HebrewRef("Psalms", second_psalm, adjusted, adjusted)]

    return [HebrewRef("Psalms", chapter, verse, verse)]


def map_lxx_to_hebrew_refs(
    book: str,
    chapter: int,
    verse: int,
    mapping_data: Dict[str, Any],
) -> List[HebrewRef]:
    norm = normalize_book_name(book)
    info = mapping_data.get(norm)
    if info is None:
        return [HebrewRef(book, chapter, verse, verse)]
    if isinstance(info, str):
        if info.strip().upper() == "NA":
            return []
        return [HebrewRef(book, chapter, verse, verse)]
    if norm == "psalms":
        return map_psalm_reference(chapter, verse, info)
    return [HebrewRef(book, chapter, verse, verse)]


CANONICAL_TO_BGT_KEY: Dict[str, str] = {
    # Pentateuch / History
    "genesis": "genesis",
    "exodus": "exodus",
    "leviticus": "leviticus",
    "numbers": "numeri",
    "deuteronomy": "deuteronomium",
    "joshua": "jozua",
    "judges": "rechters",
    "ruth": "ruth",
    "1samuel": "1samuel",
    "2samuel": "2samuel",
    "1kings": "1koningen",
    "2kings": "2koningen",
    "1chronicles": "1kronieken",
    "2chronicles": "2kronieken",
    "ezra": "ezra",
    "nehemiah": "nehemia",
    "esther": "ester",
    # Wisdom literature
    "job": "job",
    "psalms": "psalms",  # handled specially
    "proverbs": "spreuken",
    "ecclesiastes": "prediker",
    "songofsolomon": "hooglied",
    "songofsongs": "hooglied",
    # Major prophets
    "isaiah": "jesaja",
    "jeremiah": "jeremia",
    "lamentations": "klaagliederen",
    "ezekiel": "ezechil",
    "daniel": "danil",
    # Minor prophets
    "hosea": "hosea",
    "joel": "jol",
    "amos": "amos",
    "obadiah": "obadja",
    "jonah": "jona",
    "micah": "micha",
    "nahum": "nahum",
    "habakkuk": "habakuk",
    "zephaniah": "sefanja",
    "haggai": "haggai",
    "zechariah": "zacharia",
    "malachi": "maleachi",
    # Gospels and Acts
    "matthew": "mattes",
    "mark": "marcus",
    "luke": "lucas",
    "john": "johannes",
    "acts": "handelingen",
    "theacts": "handelingen",
    # Pauline epistles
    "romans": "romeinen",
    "1corinthians": "1korintirs",
    "2corinthians": "2korintirs",
    "galatians": "galaten",
    "ephesians": "efezirs",
    "philippians": "filippenzen",
    "colossians": "kolossenzen",
    "1thessalonians": "1tessalonicenzen",
    "2thessalonians": "2tessalonicenzen",
    "1timothy": "1timotes",
    "2timothy": "2timotes",
    "titus": "titus",
    "philemon": "filemon",
    "hebrews": "hebreen",
    # General epistles
    "james": "jakobus",
    "1peter": "1petrus",
    "2peter": "2petrus",
    "1john": "1johannes",
    "2john": "2johannes",
    "3john": "3johannes",
    "jude": "judas",
    # Apocalypse
    "revelation": "openbaring",
}


def resolve_bgt_book_key(book: str, chapter: int) -> Optional[str]:
    key = normalize_book_name(book)
    if key == "psalms":
        return normalize_book_name(f"Psalm {chapter}")
    return CANONICAL_TO_BGT_KEY.get(key)


def collect_bgt_entries(
    bgt_index: Dict[str, Dict[int, List[BgtEntry]]],
    ref: HebrewRef,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    book_key = resolve_bgt_book_key(ref.book, ref.chapter)
    if not book_key:
        return None, f"BGT translation not available for {ref.book}."

    chapter_map = bgt_index.get(book_key)
    if not chapter_map:
        return None, f"BGT data missing for book key '{book_key}'."

    entries = chapter_map.get(ref.chapter)
    if not entries:
        return None, f"BGT data missing for {ref.book} {ref.chapter}."

    matches = [
        entry
        for entry in entries
        if not (ref.verse_end < entry.start or ref.verse_start > entry.end)
    ]
    if not matches:
        return None, (
            f"No BGT verse covers {ref.book} {ref.chapter}:"
            f"{ref.verse_start}-{ref.verse_end}."
        )

    payload = [
        {
            "book": entry.display_book,
            "chapter": entry.chapter,
            "raw_verse": entry.raw_verse,
            "start": entry.start,
            "end": entry.end,
            "text": entry.text,
        }
        for entry in matches
    ]
    reference_summary = {
        "book": matches[0].display_book,
        "chapter": ref.chapter,
        "verse_start": ref.verse_start,
        "verse_end": ref.verse_end,
        "verses": payload,
    }
    return reference_summary, None


def fetch_bgt_exact(
    bgt_index: Dict[str, Dict[int, List[BgtEntry]]],
    mapping_data: Dict[str, Any],
    book: str,
    chapter: int,
    verse: int,
) -> Dict[str, Any]:
    refs = map_lxx_to_hebrew_refs(book, chapter, verse, mapping_data)
    if not refs:
        return {
            "available": False,
            "reason": f"No data for {book}.",
        }

    resolved = []
    for ref in refs:
        summary, error = collect_bgt_entries(bgt_index, ref)
        if error:
            return {
                "available": False,
                "reason": error,
            }
        resolved.append(summary)

    return {
        "available": True,
        "references": resolved,
    }


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

    fetch_target = max(args.top * 5, args.top + 10)
    matches = collect_similar_verses(matrix, keys, verse_lookup, query_idx, fetch_target)
    bgt_index = load_bgt_verses(args.bgt_path)
    lxx_mapping = load_lxx_mapping(args.lxx_mapping)

    query_bgt = fetch_bgt_exact(bgt_index, lxx_mapping, book, chapter, verse_number)

    for match in matches:
        match["bgt"] = fetch_bgt_exact(
            bgt_index,
            lxx_mapping,
            match["book"],
            int(match["chapter"]),
            int(match["verse"]),
        )

    query_line = [
        "# Query",
        book,
        str(chapter),
        str(verse_number),
        sanitize_text(verse_lookup[target_key]["text"]),
    ]
    print("\t".join(query_line))

    query_bgt_line = ["# Query BGT", format_bgt_summary(query_bgt)]
    print("\t".join(query_bgt_line))

    header = ["rank", "book", "chapter", "verse", "similarity", "lxx_text", "bgt"]
    print("\t".join(header))

    rows_printed = 0
    for match in matches:
        bgt_info = match.get("bgt") or {}
        if not bgt_info.get("available"):
            continue
        row = [
            str(rows_printed + 1),
            match["book"],
            str(match["chapter"]),
            str(match["verse"]),
            f"{match['similarity']:.6f}",
            sanitize_text(match["text"]),
            format_bgt_summary(bgt_info),
        ]
        print("\t".join(row))
        rows_printed += 1
        if rows_printed >= args.top:
            break

    if rows_printed == 0:
        print("# Note\tNo BGT-enabled matches found for the requested top results.")
    elif rows_printed < args.top:
        print(f"# Note\tOnly {rows_printed} matches include BGT data (requested {args.top}).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
