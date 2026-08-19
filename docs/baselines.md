# Baseline implementation boundary

All baseline classes in `src/semopkt/models/baselines.py` use the repository's
identical interaction sequences, concept-only target field, manifests, seeds,
early-stopping rule, and metric implementation. They are clean-room,
fair-input implementations of the model families needed by the experiment
matrix. They do not claim byte-for-byte identity with a third-party repository.

When a venue requires an exact official implementation, bind that external
checkout to an immutable commit, record the commit and local patch digest in
the run configuration, and export predictions in the standard schema. Do not
replace a clean-room result with a published number from another split.

The standard prediction schema permits official and clean-room runs to enter
the same paired student-level analysis without changing test targets.

