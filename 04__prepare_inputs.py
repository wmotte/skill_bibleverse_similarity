#!/usr/bin/env python3
"""Convert plain-text inputs (BGT TSV and LXX mapping JSON) into joblib blobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import gzip
import joblib
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Dutch BGT TSV and the LXX→Hebrew JSON mapping into joblib files."
    )
    root = Path(__file__).parent
    parser.add_argument(
        "--bgt-tsv",
        type=Path,
        default=root / "raw_inputs" / "bgt_with_titles.tsv",
        help="Path to the raw Bijbel in Gewone Taal TSV file.",
    )
    parser.add_argument(
        "--bgt-joblib",
        type=Path,
        default=root / "misc" / "bgt_with_titles.joblib",
        help="Destination for the serialized BGT verses.",
    )
    parser.add_argument(
        "--lxx-json",
        type=Path,
        default=root / "raw_inputs" / "lxx_to_hebrew_mapping.json",
        help="Path to the JSON file with LXX→Hebrew mapping details.",
    )
    parser.add_argument(
        "--lxx-joblib",
        type=Path,
        default=root / "misc" / "lxx_to_hebrew_mapping.joblib",
        help="Destination for the serialized mapping data.",
    )
    parser.add_argument(
        "--skip-bgt",
        action="store_true",
        help="Do not regenerate the BGT joblib file.",
    )
    parser.add_argument(
        "--skip-lxx",
        action="store_true",
        help="Do not regenerate the LXX mapping joblib file.",
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=3,
        help="Joblib compression level (default: 3).",
    )
    return parser.parse_args()


BOOK_NORMALIZER = re.compile(r"[^a-z0-9]+")


def normalize_book(name: str) -> str:
    return BOOK_NORMALIZER.sub("", (name or "").lower())


def parse_verse_range(label: str) -> tuple[int, int]:
    label = (label or "").strip()
    if not label:
        raise ValueError("Empty verse range.")
    if "-" in label:
        start, end = label.split("-", 1)
        return int(start), int(end)
    number = int(label)
    return number, number


def build_bgt_index(tsv_path: Path) -> Dict[str, Dict[int, List[Dict[str, object]]]]:
    if not tsv_path.exists():
        raise SystemExit(f"BGT TSV not found: {tsv_path}")
    index: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
    with tsv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if (row.get("type") or "").strip() != "verse":
                continue
            book = (row.get("book") or "").strip()
            chapter_raw = (row.get("chapter") or "").strip()
            verse_raw = (row.get("verse") or "").strip()
            text = (row.get("text") or "").strip()
            if not (book and chapter_raw and verse_raw and text):
                continue
            chapter = int(chapter_raw)
            start, end = parse_verse_range(verse_raw)
            book_key = normalize_book(book)
            entry = {
                "book_key": book_key,
                "display_book": book,
                "chapter": chapter,
                "raw_verse": verse_raw,
                "start": start,
                "end": end,
                "text": text,
            }
            index.setdefault(book_key, {}).setdefault(chapter, []).append(entry)
    return index


def dump_joblib(payload: object, joblib_path: Path, compress: int) -> None:
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [suffix.lower() for suffix in joblib_path.suffixes]
    if suffixes[-2:] == [".joblib", ".gz"]:
        with gzip.open(joblib_path, "wb") as handle:
            joblib.dump(payload, handle, compress=compress)
    else:
        joblib.dump(payload, joblib_path, compress=compress)


def convert_bgt(tsv_path: Path, joblib_path: Path, compress: int) -> None:
    index = build_bgt_index(tsv_path)
    dump_joblib(index, joblib_path, compress)
    print(f"Wrote {joblib_path} ({len(index)} book buckets).")


def convert_lxx(json_path: Path, joblib_path: Path, compress: int) -> None:
    if not json_path.exists():
        raise SystemExit(f"LXX JSON not found: {json_path}")
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    dump_joblib(data, joblib_path, compress)
    print(f"Wrote {joblib_path} with {len(data)} entries.")


def main() -> None:
    args = parse_args()
    if not args.skip_bgt:
        convert_bgt(args.bgt_tsv, args.bgt_joblib, args.compress)
    if not args.skip_lxx:
        convert_lxx(args.lxx_json, args.lxx_joblib, args.compress)


if __name__ == "__main__":
    main()
