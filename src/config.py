"""
Central configuration: paths, encoding maps, feature lists, and hyperparameters.
All other modules import from here.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/bank_marketing.csv"
MODEL_DIR        = "models/"
REPORTS_DIR      = "reports/"
FIGURES_DIR      = "reports/figures/"

RF_MODEL_PATH    = "models/random_forest.pkl"
LR_MODEL_PATH    = "models/logistic_regression.pkl"
SCALER_PATH      = "models/scaler.pkl"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET = "term_deposit"

# ── Encoding maps ─────────────────────────────────────────────────────────────
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
DAY_MAP    = {"mon": 2, "tue": 3, "wed": 4, "thu": 5, "fri": 6}
BINARY_MAP = {"yes": 1, "no": 0, "unknown": -1}
TARGET_MAP = {"yes": 1, "no": 0}

# ── Outlier capping thresholds (IQR-derived) ──────────────────────────────────
OUTLIER_CAPS = {
    "age":      69.5,
    "duration": 644.5,
    "campaign": 6.0,
}

# ── Education consolidation ───────────────────────────────────────────────────
# Merge all "basic.*y" categories into one
BASIC_EDUCATION_VARIANTS = ["basic.9y", "basic.4y", "basic.6y"]
BASIC_EDUCATION_LABEL    = "basic.school"

# ── Features to drop after feature selection (low mutual info) ────────────────
LOW_INFO_FEATURES = [
    "day_of_week", "d_unknownm", "d_self-employed", "d_entrepreneur",
    "d_housemaid", "d_management", "d_services", "d_technician",
    "d_unemployed", "d_unknownj", "d_university.degree", "d_unknowne",
    "d_illiterate", "d_retired", "d_blue-collar", "d_single",
    "d_student", "d_professional.course", "d_high.school", "d_married",
    "d_telephone", "d_nonexistent",
]

# ── Columns to standardize (continuous numeric only) ─────────────────────────
SCALE_COLS = [
    "age", "duration", "campaign", "previous",
    "employment_variation_rate", "consumer_price_index",
    "consumer_confidence_index", "3_months_euribor_rate",
    "number_of_employees_in_the_bank",
]

# ── Train/test split ──────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Random Forest (tuned via RandomizedSearchCV) ──────────────────────────────
RF_PARAMS = {
    "n_estimators":    100,
    "max_features":   "sqrt",
    "max_depth":       10,
    "min_samples_split": 2,
    "min_samples_leaf":  1,
    "bootstrap":       True,
    "random_state":    RANDOM_STATE,
}

# ── Logistic Regression (tuned via GridSearchCV) ──────────────────────────────
LR_PARAMS = {
    "C":            0.3907,
    "penalty":      "l2",
    "random_state": RANDOM_STATE,
    "max_iter":     1000,
}
