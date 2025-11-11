# Bible Verse Similarity Skill

A Claude Skill for finding semantically similar Bible verses using pre-computed embeddings.

## Installation

### For Claude Desktop

1. **Copy the skill folder** to your Claude Desktop skills directory:

   **macOS/Linux**:
   ```bash
   cp -r bible-verse-similarity ~/Library/Application\ Support/Claude/skills/
   ```

   **Windows**:
   ```powershell
   Copy-Item -Recurse bible-verse-similarity "$env:APPDATA\Claude\skills\"
   ```

2. **Restart Claude Desktop** to load the new skill.

3. **Verify installation** by asking Claude:
   ```
   "What skills do you have available?"
   ```

### For Claude Code

When using Claude Code from this repository, the skill will automatically be available since it's in the project directory.

## Usage

Once installed, you can ask Claude natural language queries like:

- "Find verses similar to John 3:16"
- "Show me passages related to Psalm 23:1"
- "What verses have similar meaning to Romans 8:28?"
- "Give me 15 verses similar to Matthew 5:14"

Claude will automatically invoke this skill and present the results in a readable format.

## Requirements

### Data Files
The skill requires pre-built data files in the `misc/` directory:
- `verse_similarity_bundle_int8.joblib` (semantic embeddings)
- `bgt_with_titles.joblib` (Dutch BGT translation)
- `lxx_to_hebrew_mapping.joblib` (LXX↔Hebrew mappings)

### Python Dependencies
- Python 3.7+
- numpy
- joblib

Install dependencies:
```bash
pip install numpy joblib
```

## Features

- **Semantic Search**: Finds verses by meaning, not just keywords
- **LXX & New Testament**: Searches Greek Septuagint and NT texts
- **Dutch Translation**: Includes BGT (Bijbel in Gewone Taal) translations
- **Smart Mapping**: Handles LXX-to-Hebrew verse number differences
- **Flexible Input**: Accepts various reference formats (John 3:16, John 3 16, etc.)
- **Ranked Results**: Sorts by similarity score (0-1 scale)

## Data Sources

- **LXX**: Septuagint (Greek Old Testament)
- **NT**: Greek New Testament
- **BGT**: Dutch "Bijbel in Gewone Taal" translation
- **Embeddings**: Generated via Ollama with BGE-M3 model

## Coverage

The skill searches:
- All LXX books (Genesis through Malachi in Septuagint form)
- New Testament (Matthew through Revelation)
- Deuterocanonical books included in LXX (Wisdom, Sirach, etc.)

**Note**: Old Testament uses LXX (Greek) text and numbering, which may differ from Hebrew Bible versification.

## Limitations

- Only verses in the LXX/NT dataset can be queried
- Results are filtered to show only verses with BGT translations
- Similarity is based on Greek text, not English translations
- Psalm numbering follows LXX conventions (e.g., LXX Psalm 9-10 ≠ Hebrew Psalm 9-10)

## Troubleshooting

### "Unknown book name"
The script will suggest alternatives. Use full book names or standard abbreviations.

### "Verse not found"
- Verify the verse exists in the LXX/NT corpus
- Check for versification differences (LXX vs Hebrew)
- Ensure chapter and verse numbers are correct

### No results returned
Some verses may not have BGT translations. The script automatically filters to include only verses with translations. Try increasing `--top` to get more candidates.

## Advanced Usage

You can also invoke the script directly:

```bash
python3 03__query_with_bgt.py --reference "John 3:16" --top 10
```

Options:
- `--reference "VERSE"`: Bible reference (John 3:16, Genesis 1:1, etc.)
- `--top N`: Number of results to return (default: 10)
- `--bundle-path PATH`: Custom path to embeddings bundle
- `--bgt-path PATH`: Custom path to BGT translation data
- `--lxx-mapping PATH`: Custom path to LXX mapping data

Positional arguments also work:
```bash
python3 03__query_with_bgt.py John 3 16 --top 5
```

## License

See LICENSE file in the repository root.

## Author

Wim Otte (w.m.otte@umcutrecht.nl)
