# Semantic-control resources

Human-verified paraphrases and external definitions are data artifacts rather
than executable code. For each dataset, provide the following UTF-8 files:

- `data/text/<dataset>_paraphrases.csv`
- `data/text/<dataset>_normalizations.csv`
- `data/text/<dataset>_definitions.csv`

Each file has exactly two required columns: `kc_text_norm` and
`replacement_text`. Every row is joined to the normalized descriptor before
embedding. The code records the resulting text-set hash in the embedding-cache
metadata. Missing files stop the corresponding E10 condition; they are never
silently synthesized.

History-level synonym robustness uses
`data/text/<dataset>_synonyms.csv` with the columns `source_token` and
`replacement_token`. Source tokens must be unique after case folding. The
same deterministic token-selection seed is used for every compared model.

The anonymous archive should include these files only after verifying that
their metadata and free-text cells contain no annotator identity. Raw
annotation logs and reviewer names are not part of the release.
