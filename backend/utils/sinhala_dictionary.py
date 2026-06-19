import pandas as pd
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_CLASS_MAP = {}

def load_dictionary(csv_path="data/splits/sinhala_word_map.csv"):
    global _CLASS_MAP
    try:
        df = pd.read_csv(csv_path)
        _CLASS_MAP = dict(zip(df["class_id"], df["class_name_english"]))
        logger.info(f"Loaded {len(_CLASS_MAP)} words into dictionary.")
    except Exception as e:
        logger.error(f"Failed to load dictionary: {e}")
        _CLASS_MAP = {}

def get_word(class_id: int) -> str:
    return _CLASS_MAP.get(class_id, f"Class {class_id}")

def get_full_dictionary() -> dict:
    return _CLASS_MAP
