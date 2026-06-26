import csv
from pathlib import Path
from utils.logger import logger

class SinhalaDictionary:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.word_map = {}
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            logger.warning(f"Sinhala word map not found at {self.csv_path}")
            return
            
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_id = int(row['class_id'])
                    self.word_map[class_id] = {
                        'sinhala': row['class_name_sinhala'],
                        'english': row['class_name_english']
                    }
            logger.info(f"Loaded {len(self.word_map)} Sinhala words from dictionary.")
        except Exception as e:
            logger.error(f"Failed to load Sinhala dictionary: {e}")

    def get_word(self, class_id: int) -> dict:
        return self.word_map.get(class_id, {
            'sinhala': f"Class_{class_id}",
            'english': f"Class_{class_id}"
        })
