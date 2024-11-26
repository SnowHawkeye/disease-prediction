import os

RANDOM_SEED = 39
PROCESSED_DATA_PATH = "../out/data.pickle"

# Get the absolute path to the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define SOURCE_ROOT relative to the script directory
SOURCE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../src"))

# Define DATA_ROOT as a subdirectory of SOURCE_ROOT
DATA_ROOT = os.path.join(SOURCE_ROOT, "features/mimic/scripts/data/34-prediction-performance/categorized_analyses")

ALL_LAB_CATEGORIES = [
    "hematology",
    "metabolic",
    "renal",
    "hepatology",
    "infectiology",
    "nutrition",
    "toxicology",
    "cardiology",
    "endocrine",
    "muscular",
    "immunology_inflammation",
    "tumor_marker",
    "reproduction",
    "body_fluids",
    "pulmonary",
    "hepatic_renal",
]