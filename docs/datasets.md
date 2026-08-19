# Dataset preparation

Raw data are deliberately excluded. Place provider files under
`data/raw/NIPS34`, `data/raw/Algebra05`, and `data/raw/Assist09`. The
preprocessor resolves documented column aliases, preserves source row IDs,
retains genuine retries, removes only exact duplicate rows, removes students
with fewer than six interactions, and keeps each student's earliest 50 rows.

The standardized schema is:

`dataset, student_id, question_id, kc_id, kc_text_raw, kc_text_norm,
kc_components, correct, timestamp, position, source_row_id`.

Multiple KC labels are split with the configured source delimiters, normalized
without changing their source order, and joined with ` [SEP] `. A held-out
interaction is any interaction whose component set intersects the held-out KC
set. Question identifiers are retained for auditing but are not model inputs in
the concept-only fair-input condition. Algebra05 uses the configured composite
`Problem Name :: Step Name` item key. If a timestamp alias is absent, source-row
order is used directly; a question identifier is never substituted as time.

NIPS34 preparation additionally requires `answer_metadata_task_3_4.csv`,
`question_metadata_task_3_4.csv`, and `subject_metadata.csv`. The primary
interaction table is joined on answer and question IDs with many-to-one
validation; only level-3 subjects form the canonical KC descriptor. Every
participating input file and the join rule are recorded in the preprocessing
audit.

The command below writes a standardized table, statistics JSON, and SHA-256
digest:

```bash
python scripts/prepare_data.py \
  --config configs/data/assist09.yaml \
  --raw-root data/raw/Assist09 \
  --output data/processed/Assist09/interactions.parquet
```

Raw statistics are compared with the public reference counts. A mismatch is a
hard error unless `--allow-statistic-mismatch` is explicitly supplied and the
resulting audit record is retained.
