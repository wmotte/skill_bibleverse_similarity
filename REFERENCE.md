# Bible Verse Similarity - Technical Reference

## Architecture Overview

### Embedding System

The similarity search is powered by semantic embeddings generated using the BGE-M3 model via Ollama. Each verse is represented as a high-dimensional vector (typically 1024 dimensions), and similarity is computed using cosine similarity (dot product of normalized vectors).

### Data Pipeline

```
Raw Input (TSV/JSON)
    ↓
04__prepare_inputs.py
    ↓
Joblib Binary Files
    ↓
00__make_similarity.py (with Ollama)
    ↓
Full Precision Bundle (float32)
    ↓
02__compress_bundle.py
    ↓
Quantized Bundle (int8/float16)
    ↓
03__query_with_bgt.py
    ↓
Similar Verses with Translations
```

## Verse Key System

All verses are indexed using a canonical key format:
```
{canonical_book_name}|{chapter}|{verse}
```

Example: `"John|3|16"`

### Book Name Normalization

Book names undergo normalization to handle variations:
1. Convert to lowercase
2. Remove all non-alphanumeric characters (spaces, periods, hyphens)
3. Match against canonical names and abbreviations

Examples:
- "1 Samuel" → "1samuel"
- "Song of Solomon" → "songofsolomon"
- "Rom." → "romans"

When an exact match isn't found, the system uses fuzzy matching (`difflib.get_close_matches`) with a 0.6 cutoff to suggest corrections.

## LXX to Hebrew Mapping

The LXX (Septuagint) and Hebrew Bible have different verse numbering schemes, particularly in:

### Psalms
LXX Psalms 9-10 are merged into Hebrew Psalm 9 and 10. The mapping uses:
- `PSALM_BOUNDARIES`: Defines split points for merged psalms
- `detailed_mapping`: JSON structure with per-chapter mappings

Example:
- LXX Psalm 9:1-21 → Hebrew Psalm 9:1-21
- LXX Psalm 10:1-18 → Hebrew Psalm 10:1-18
- LXX Psalm 146 → Hebrew Psalm 147:1-11
- LXX Psalm 147 → Hebrew Psalm 147:12-20

### Kingdom Books
The LXX naming differs for historical books:
- 1 Samuel = 1 Kingdoms (LXX)
- 2 Samuel = 2 Kingdoms (LXX)
- 1 Kings = 3 Kingdoms (LXX)
- 2 Kings = 4 Kingdoms (LXX)

These mappings are handled by `CANONICAL_TO_BGT_KEY` in `03__query_with_bgt.py`.

## BGT Translation System

BGT (Bijbel in Gewone Taal) uses Dutch book names and special handling for Psalms.

### Psalm Handling
Each psalm is a separate "book" in BGT:
- Psalm 1 → "psalm1"
- Psalm 23 → "psalm23"
- Psalm 119 → "psalm119"

The `resolve_bgt_book_key()` function detects the book "Psalms" and converts it to the chapter-specific book key.

### Verse Range Support
BGT verses can span multiple verse numbers (e.g., "1-2" for merged verses). The system:
1. Parses range strings: "5" → (5, 5), "3-4" → (3, 4)
2. Checks overlap: Query verse [start, end] overlaps with BGT entry [start, end]
3. Returns all overlapping entries

## Quantization

The default bundle uses int8 quantization to reduce file size by ~75% with minimal accuracy loss.

### Symmetric Int8 Quantization
```python
scale = 127.0 / max(abs(embeddings))
quantized = round(embeddings * scale).astype(int8)

# Dequantization at query time:
float_embeddings = quantized.astype(float32) / scale
```

Metadata structure:
```json
{
  "quantization": {
    "mode": "symmetric_int8",
    "scale": 127.43892,
    "max_abs": 0.9956473,
    "apply": "matrix_int8 / scale"
  }
}
```

## Similarity Computation

### Algorithm
1. Dequantize the bundle (if quantized)
2. Extract query vector: `query_vec = matrix[query_index]`
3. Compute dot products: `similarities = matrix @ query_vec`
4. Exclude self: `similarities[query_index] = -1.0`
5. Use partial sort: `np.argpartition(similarities, -N)[-N:]`
6. Full sort top-N: `np.argsort(similarities[candidates])[::-1]`
7. Deduplicate by key (handles duplicate verses)
8. Filter to verses with BGT data available
9. Return top-N results

### Optimization
The script requests `top * 5` candidates initially because:
- Some verses may not have BGT translations
- Deduplication may remove entries
- Ensures at least `top` results with translations

## Output Format

### TSV Structure
```
# Query	{book}	{chapter}	{verse}	{lxx_text}
# Query BGT	{bgt_summary}
rank	book	chapter	verse	similarity	lxx_text	bgt
1	Romans	5	8	0.856234	{lxx_text}	{bgt_reference} | {bgt_text}
2	1 John	4	9	0.842156	{lxx_text}	{bgt_reference} | {bgt_text}
```

### BGT Summary Format
For verses that map to multiple BGT entries (e.g., merged verses):
```
{book} {chapter}:{verse} {text} | {book} {chapter}:{verse} {text}
```

Multiple references are separated by ` || `.

## Error Handling

All errors use `SystemExit` with descriptive messages:
- **Missing files**: "Bundle file not found: {path}"
- **Invalid bundle**: "Bundle does not contain expected data. Rebuild the bundle."
- **Unknown book**: "Unknown book name '{book}'. Please provide a valid name or abbreviation."
- **Ambiguous book**: "Ambiguous book name '{book}'. Did you mean: {suggestions}?"
- **Verse not found**: "Verse not found in bundle metadata: {key}"
- **Invalid quantization**: "Bundle stores int8 data without quantization metadata."

Scripts exit with code 130 on KeyboardInterrupt for clean signal handling.

## Performance Characteristics

### Bundle Loading
- float32: ~5s for 30k verses (120MB)
- int8: ~2s for 30k verses (32MB)

### Query Time
- Index lookup: O(1)
- Similarity computation: O(n*d) where n=verses, d=dimensions
- Partial sort: O(n) average
- Full sort of candidates: O(k log k) where k=top*5

Typical query: ~0.5-1 second for 30k verses on modern hardware.

### Memory Usage
- float32 bundle: ~1GB RAM
- int8 bundle: ~300MB RAM
- BGT index: ~50MB RAM
- LXX mapping: ~1MB RAM

## File Formats

### Bundle (.joblib)
```python
{
    "matrix": np.ndarray,  # shape: (num_verses, embedding_dim)
    "keys": List[str],     # verse keys: "Book|Chapter|Verse"
    "verses": List[dict],  # full metadata per verse
    "meta": {
        "embedding_model": str,
        "quantization": dict | None
    }
}
```

### BGT Index (.joblib)
```python
{
    "book_key": {
        chapter_num: [
            {
                "book_key": str,
                "display_book": str,
                "chapter": int,
                "raw_verse": str,
                "start": int,
                "end": int,
                "text": str
            }
        ]
    }
}
```

### LXX Mapping (.joblib)
```python
{
    "book_name": {
        "detailed_mapping": {
            "lxx_chapter": {
                "hebrew_psalm": int | str,
                "verses": dict | None
            }
        }
    } | "NA" | None
}
```
