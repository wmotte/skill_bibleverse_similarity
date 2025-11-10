#!/usr/bin/env python3
#
# Wim Otte (w.m.otte@umcutrecht.nl)
"""Down-sample the similarity bundle matrix to a lower precision representation.

The default behavior rewrites the embeddings matrix to float16 while preserving
the rest of the bundle, resulting in a significantly smaller serialized file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a similarity bundle to a lower-precision matrix."
    )
    default_source = Path(__file__).parent / "artifacts" / "verse_similarity_bundle.joblib"
    default_target = (
        Path(__file__).parent / "artifacts" / "verse_similarity_bundle_fp16.joblib"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help=f"Input bundle path (default: {default_source}).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=default_target,
        help=f"Output bundle path (default: {default_target}).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Target matrix dtype (float16, int8).",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=None,
        help="Optional number of decimals to retain before changing dtype.",
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=3,
        help="Joblib compression level (default: 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file when it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source bundle not found: {args.source}")
    if args.target.exists() and not args.force:
        raise SystemExit(
            f"Target bundle already exists: {args.target}. Use --force to overwrite."
        )

    payload: Dict[str, Any] = joblib.load(args.source)
    if "matrix" not in payload:
        raise SystemExit("Source bundle does not contain a 'matrix' entry.")

    matrix = payload["matrix"]
    if not isinstance(matrix, np.ndarray):
        raise SystemExit("Source bundle matrix must be a NumPy array.")

    target_dtype = np.dtype(args.dtype)
    quantization: Optional[Dict[str, Any]] = None
    work_matrix = matrix.astype(np.float32, copy=False)
    if args.round_decimals is not None:
        work_matrix = np.round(work_matrix, args.round_decimals)

    if np.issubdtype(target_dtype, np.floating):
        converted = work_matrix.astype(target_dtype)
    elif np.issubdtype(target_dtype, np.signedinteger):
        if target_dtype != np.int8:
            raise SystemExit(f"Only int8 quantization is supported (requested {target_dtype}).")
        max_abs = float(np.abs(work_matrix).max())
        if max_abs == 0:
            max_abs = 1.0
        scale = 127.0 / max_abs
        converted = np.round(work_matrix * scale).astype(np.int8)
        quantization = {
            "mode": "symmetric_int8",
            "scale": scale,
            "max_abs": max_abs,
            "apply": "matrix_int8 / scale",
        }
    else:
        raise SystemExit(f"Unsupported dtype request: {target_dtype}")

    payload = dict(payload)
    payload["matrix"] = converted
    meta: Dict[str, Any] = dict(payload.get("meta") or {})
    meta.update(
        {
            "compressed_from": os.fspath(args.source),
            "matrix_dtype": str(converted.dtype),
            "rounded_decimals": args.round_decimals,
        }
    )
    if quantization:
        meta["quantization"] = quantization
    else:
        meta.pop("quantization", None)
    payload["meta"] = meta

    joblib.dump(payload, args.target, compress=args.compress)

    summary = {
        "source_path": os.fspath(args.source),
        "target_path": os.fspath(args.target),
        "source_dtype": str(matrix.dtype),
        "target_dtype": str(converted.dtype),
        "round_decimals": args.round_decimals,
        "source_shape": list(matrix.shape),
        "quantization": quantization,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
