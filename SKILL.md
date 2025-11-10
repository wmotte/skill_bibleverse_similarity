---
name: bible-verse-similarity
description: Find semantically similar Bible verses using embeddings. Use this when users want to discover verses with related themes, concepts, or teachings to a given reference. Searches LXX and New Testament, with Dutch BGT translation mappings.
---

# Bible Verse Similarity Search

This Skill enables semantic search across Bible verses to find passages with similar meanings, themes, or concepts. It uses pre-computed embeddings and provides results enriched with Dutch "Bijbel in Gewone Taal" (BGT) translations.

## When to Use This Skill

Use this Skill when:
- User asks to find verses similar to a specific Bible reference
- User wants to discover related passages or parallel themes
- User requests verses that echo or relate to a given verse
- User wants cross-references based on semantic meaning (not just keyword matching)

## How It Works

The Skill queries a pre-built similarity index based on the `03__query_with_bgt.py` script. The index contains semantic embeddings of LXX (Septuagint) and New Testament verses.

## Instructions

### Step 1: Parse the User's Request

Extract the Bible verse reference from the user's query. The reference can be in multiple formats:
- "John 3:16"
- "John 3 16"
- "Genesis 1.1"
- Book name with chapter and verse separated by spaces, colons, or periods

### Step 2: Execute the Query

Run the similarity search using the Bash tool:

```bash
python3 03__query_with_bgt.py --reference "VERSE_REFERENCE" --top N
```

Replace:
- `VERSE_REFERENCE` with the parsed reference (e.g., "John 3:16")
- `N` with the number of results requested (default: 10)

Alternative invocation using positional arguments:
```bash
python3 03__query_with_bgt.py BOOK CHAPTER VERSE --top N
```

### Step 3: Parse and Present Results

The script outputs TSV format with:
- Line 1: Query verse information (# Query, book, chapter, verse, LXX text)
- Line 2: Query verse BGT translation (# Query BGT)
- Line 3: Header row (rank, book, chapter, verse, similarity, lxx_text, bgt)
- Subsequent lines: Similar verses ranked by semantic similarity

Present the results in a readable format:
1. Show the query verse with its text
2. List similar verses with:
   - Rank and similarity score (0-1 scale, higher = more similar)
   - Book, chapter, verse reference
   - LXX text
   - BGT (Dutch) translation

### Step 4: Handle Errors

Common errors and solutions:
- **"Unknown book name"**: The script suggests alternatives. Re-run with corrected book name.
- **"Verse not found in bundle"**: The verse isn't in the LXX/NT dataset (Old Testament uses LXX, not Hebrew).
- **"Bundle file not found"**: Ensure `misc/verse_similarity_bundle_int8.joblib` exists.

## Examples

### Example 1: Basic Query
**User**: "Find verses similar to John 3:16"

**Action**:
```bash
python3 03__query_with_bgt.py --reference "John 3:16" --top 10
```

**Response Format**:
"Here are the top 10 verses most similar to John 3:16:

Query: John 3:16 - [LXX text]
BGT: [Dutch translation]

Similar verses:
1. Romans 5:8 (similarity: 0.856) - [text and BGT]
2. 1 John 4:9 (similarity: 0.842) - [text and BGT]
..."

### Example 2: Custom Result Count
**User**: "Show me 5 verses similar to Psalm 23:1"

**Action**:
```bash
python3 03__query_with_bgt.py --reference "Psalm 23:1" --top 5
```

### Example 3: Using Positional Arguments
**User**: "Find related verses to Ruth 1:16"

**Action**:
```bash
python3 03__query_with_bgt.py Ruth 1 16 --top 10
```

## Notes

- The similarity scores range from 0 to 1, where higher values indicate greater semantic similarity
- The query verse itself is excluded from results
- Only verses with available BGT translations are included in the output
- The system uses LXX for Old Testament (not Hebrew Bible), so verse numbering may differ from standard English Bibles
- Psalm numbering follows LXX conventions and is automatically mapped to Hebrew numbering for BGT lookup

## Technical Details

**Data Sources**:
- LXX (Septuagint) and New Testament Greek text
- Dutch "Bijbel in Gewone Taal" translation
- LXX-to-Hebrew verse mapping for proper cross-referencing

**Required Files** (all relative to repository root):
- `misc/verse_similarity_bundle_int8.joblib`: Pre-computed embeddings (int8 quantized)
- `misc/bgt_with_titles.joblib.gz`: Indexed BGT translation
- `misc/lxx_to_hebrew_mapping.joblib`: LXX→Hebrew verse mappings

**Script Dependencies**:
- Python 3.7+
- numpy, joblib, standard library (csv, json, re, difflib)
