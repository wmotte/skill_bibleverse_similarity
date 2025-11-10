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
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD


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
        help="Target matrix dtype (float16, int8, svd-int8).",
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
        "--svd-components",
        type=int,
        default=64,
        help="Number of SVD components when dtype is svd-int8 (default: 64).",
    )
    parser.add_argument(
        "--svd-random-state",
        type=int,
        default=0,
        help="Random seed for TruncatedSVD when dtype is svd-int8.",
    )
    parser.add_argument(
        "--svd-projection-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Stored dtype for the SVD projection matrix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target file when it already exists.",
    )
    return parser.parse_args()


def quantize_per_dim_symmetric_int8(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize each column independently to int8 using symmetric scaling."""
    max_abs = np.max(np.abs(data), axis=0)
    scales = np.maximum(max_abs / 127.0, 1e-12)
    quantized = np.round(data / scales).astype(np.int8)
    return quantized, scales.astype(np.float32)


def compress_matrix_with_svd_int8(
    matrix: np.ndarray,
    components: int,
    random_state: int,
    projection_dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Return an SVD-reduced, per-dimension quantized matrix plus metadata."""
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    reduced = svd.fit_transform(matrix)  # Equivalent to U * Sigma.
    quantized, scales = quantize_per_dim_symmetric_int8(reduced)
    projection = svd.components_.astype(np.float32, copy=True)
    if projection_dtype == "float16":
        projection = projection.astype(np.float16)
    meta = {
        "format": "svd_int8",
        "components": components,
        "random_state": random_state,
        "explained_variance_ratio": float(np.sum(svd.explained_variance_ratio_)),
        "projection_dtype": str(projection.dtype),
        "quantization": {
            "mode": "per_component_symmetric_int8",
            "scales_dtype": "float32",
        },
    }
    return quantized, scales, projection, meta


def restore_svd_int8_matrix(
    encoded: np.ndarray,
    payload: Dict[str, Any],
) -> np.ndarray:
    projection = payload.get("projection")
    scales = payload.get("svd_scales")
    if projection is None or scales is None:
        raise SystemExit(
            "Source bundle stores matrix_format=svd_int8 but lacks projection/scales."
        )
    projection_arr = np.asarray(projection, dtype=np.float32)
    scales_arr = np.asarray(scales, dtype=np.float32)
    if encoded.ndim != 2:
        raise SystemExit("Compressed matrix must be 2-dimensional.")
    if projection_arr.shape[0] != encoded.shape[1] or scales_arr.shape[0] != encoded.shape[1]:
        raise SystemExit("SVD metadata mismatch detected in the source bundle.")
    reduced = encoded.astype(np.float32) * scales_arr
    return reduced @ projection_arr


def to_float32_matrix(
    matrix: np.ndarray,
    payload: Dict[str, Any],
) -> np.ndarray:
    """Return a float32 view of the source matrix, decoding quantization if present."""
    meta = payload.get("meta") or {}
    matrix_format = meta.get("matrix_format")
    working = matrix
    if matrix_format == "svd_int8":
        working = restore_svd_int8_matrix(working, payload)
    if not isinstance(working, np.ndarray):
        raise SystemExit("Source bundle matrix must be a NumPy array.")
    if working.dtype == np.float32:
        return working
    if working.dtype == np.float16:
        return working.astype(np.float32)
    if working.dtype == np.int8:
        quant_meta = meta.get("quantization") if isinstance(meta, dict) else None
        if not quant_meta or quant_meta.get("mode") != "symmetric_int8":
            raise SystemExit(
                "Int8 source matrix is missing symmetric quantization metadata. "
                "Rebuild the float32 bundle first."
            )
        scale = float(quant_meta.get("scale") or 0.0)
        if scale <= 0:
            raise SystemExit("Invalid quantization scale detected in source bundle.")
        return working.astype(np.float32) / scale
    return working.astype(np.float32)


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

    quantization: Optional[Dict[str, Any]] = None
    svd_meta: Optional[Dict[str, object]] = None
    target_dtype = None if args.dtype == "svd-int8" else np.dtype(args.dtype)
    work_matrix = to_float32_matrix(matrix, payload)
    if args.round_decimals is not None:
        work_matrix = np.round(work_matrix, args.round_decimals)

    if args.dtype == "svd-int8":
        if args.svd_components < 1:
            raise SystemExit("--svd-components must be a positive integer.")
        (
            converted,
            svd_scales,
            projection,
            svd_meta,
        ) = compress_matrix_with_svd_int8(
            work_matrix,
            args.svd_components,
            args.svd_random_state,
            args.svd_projection_dtype,
        )
        payload["projection"] = projection
        payload["svd_scales"] = svd_scales
    elif np.issubdtype(target_dtype, np.floating):
        converted = work_matrix.astype(target_dtype)
        payload.pop("projection", None)
        payload.pop("svd_scales", None)
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
        payload.pop("projection", None)
        payload.pop("svd_scales", None)
    else:
        raise SystemExit(f"Unsupported dtype request: {target_dtype}")

    payload = dict(payload)
    payload["matrix"] = converted
    meta: Dict[str, Any] = dict(payload.get("meta") or {})
    matrix_format = svd_meta["format"] if svd_meta else str(converted.dtype)
    meta.update(
        {
            "compressed_from": os.fspath(args.source),
            "matrix_dtype": str(converted.dtype),
            "matrix_format": matrix_format,
            "rounded_decimals": args.round_decimals,
        }
    )
    if svd_meta:
        meta["svd"] = svd_meta
        meta["quantization"] = {
            "mode": "svd_int8",
            "detail": "per-component symmetric int8 scales stored in payload['svd_scales']",
        }
    elif quantization:
        meta["quantization"] = quantization
    else:
        meta.pop("quantization", None)
        meta.pop("svd", None)
    payload["meta"] = meta

    joblib.dump(payload, args.target, compress=args.compress)

    summary = {
        "source_path": os.fspath(args.source),
        "target_path": os.fspath(args.target),
        "source_dtype": str(matrix.dtype),
        "target_dtype": str(converted.dtype),
        "matrix_format": matrix_format,
        "round_decimals": args.round_decimals,
        "source_shape": list(matrix.shape),
        "quantization": quantization,
        "svd": svd_meta,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
