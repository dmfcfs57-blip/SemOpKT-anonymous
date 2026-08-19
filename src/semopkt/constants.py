"""Repository-wide constants with no machine-specific state."""

PACKAGE_VERSION = "1.0.0"
PREPROCESS_VERSION = "semopkt-preprocess-v2"
STANDARD_COLUMNS = (
    "dataset",
    "student_id",
    "question_id",
    "kc_id",
    "kc_text_raw",
    "kc_text_norm",
    "kc_components",
    "correct",
    "timestamp",
    "position",
    "source_row_id",
)
PREDICTION_COLUMNS = (
    "dataset",
    "experiment",
    "protocol",
    "model",
    "seed",
    "student_id",
    "question_id",
    "kc_id",
    "correct",
    "probability",
    "position",
    "seen_status",
    "calibration_status",
)
DEFAULT_SEEDS = tuple(range(202601, 202611))
TRAIN_SIZES = (8, 16, 32, 64)
HISTORY_WINDOWS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 50))
