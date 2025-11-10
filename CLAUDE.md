# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bible verse similarity search system that uses semantic embeddings to find similar verses across biblical texts. The project includes tools for generating embeddings, querying similarities, and mapping between different Bible translations (LXX to Hebrew, and Dutch BGT translation).

## Core Architecture

### Pipeline Workflow

The system follows a multi-stage pipeline:

1. **Data Preparation** (`04__prepare_inputs.py`): Converts raw TSV/JSON inputs into compressed joblib files
2. **Embedding Generation** (`00__make_similarity.py`): Generates semantic embeddings via Ollama API
3. **Bundle Compression** (`02__compress_bundle.py`): Downcasts embeddings to float16/int8 for storage efficiency
4. **Querying**: Two query interfaces available:
   - `01__query_from_similarity.py`: Basic similarity search (returns JSON)
   - `03__query_with_bgt.py`: Enhanced search with Dutch BGT translation mapping (returns TSV)

### Key Technical Components

**Verse Key Format**: All verses are indexed using the format `{book}|{chapter}|{verse}` (see `make_key()` function across all scripts).

**Book Name Normalization**: The system normalizes book names by stripping non-alphanumeric characters and lowercasing (via `normalize_book_name()`). Uses fuzzy matching with `difflib.get_close_matches()` for user input.

**LXX to Hebrew Mapping**: Special handling for Psalms where LXX and Hebrew numbering diverge. The mapping supports:
- Simple chapter redirects
- Range mappings with verse adjustments
- Merged psalm handling (e.g., Psalms 9-10, 114-115)

**BGT Translation Mapping**: Maps canonical English book names to Dutch BGT book keys via `CANONICAL_TO_BGT_KEY`. Psalms are handled specially by chapter number (each psalm is a separate book in BGT).

**Quantization Support**: Embeddings can be stored as float32, float16, or int8. Int8 uses symmetric quantization (scale = 127/max_abs) with metadata stored in bundle["meta"]["quantization"].

## Common Commands

### Generate embeddings from scratch
```bash
python3 00__make_similarity.py --rebuild --data-path raw_inputs/lxx_and_nt.tsv --bundle-path artifacts/verse_similarity_bundle.joblib
```

### Compress an existing bundle
```bash
# To float16 (default)
python3 02__compress_bundle.py --source artifacts/verse_similarity_bundle.joblib --target artifacts/verse_similarity_bundle_fp16.joblib

# To int8 (smaller but lossy)
python3 02__compress_bundle.py --source artifacts/verse_similarity_bundle.joblib --target artifacts/verse_similarity_bundle_int8.joblib --dtype int8
```

### Query for similar verses (JSON output)
```bash
# All three reference formats work:
python3 01__query_from_similarity.py --reference "John 3:16" --top 10
python3 01__query_from_similarity.py --book John --chapter 3 --verse 16 --top 10
python3 01__query_from_similarity.py John 3 16 --top 10
```

### Query with BGT translation (TSV output)
```bash
python3 03__query_with_bgt.py --reference "John 3:16" --top 10
# Uses default paths in misc/ directory:
# - verse_similarity_bundle_int8.joblib
# - bgt_with_titles.joblib
# - lxx_to_hebrew_mapping.joblib
```

### Prepare input data files
```bash
python3 04__prepare_inputs.py --bgt-tsv raw_inputs/bgt_with_titles.tsv --lxx-json raw_inputs/lxx_to_hebrew_mapping.json
```

## Data Files

### Input Data
- `raw_inputs/lxx_and_nt.tsv`: Source text for LXX and New Testament verses
- `raw_inputs/bgt_with_titles.tsv`: Dutch Bijbel in Gewone Taal translation
- `raw_inputs/lxx_to_hebrew_mapping.json`: LXX to Hebrew verse number mappings

### Generated Artifacts
- `misc/verse_similarity_bundle_int8.joblib`: Quantized embedding bundle (default for queries)
- `misc/bgt_with_titles.joblib`: Indexed BGT verses for fast lookup
- `misc/lxx_to_hebrew_mapping.joblib`: Compiled mapping data

### Bundle Structure
Joblib bundles contain:
- `matrix`: NumPy array of embeddings (shape: [num_verses, embedding_dim])
- `keys`: List of verse keys matching matrix rows
- `verses`: List of dicts with full verse metadata (book, chapter, verse, text, etc.)
- `meta`: Dict with embedding model info and optional quantization parameters

## Dependencies

The project requires:
- `numpy`: Matrix operations and embeddings
- `joblib`: Serialization/deserialization
- `difflib`: Fuzzy book name matching
- Ollama API running locally (default: http://localhost:11434) with embedding model (default: bge-m3:latest)

No requirements.txt or pyproject.toml exists; dependencies must be installed manually via pip.

## Script Execution Notes

All scripts are executable and include proper signal handling for KeyboardInterrupt (exits with code 130). They use `argparse` with sensible defaults based on `Path(__file__).parent` for portability.

The `00__make_similarity.py` script supports concurrent embedding generation via `--concurrency` (default: 4 threads) and includes progress reporting every 250 verses.
