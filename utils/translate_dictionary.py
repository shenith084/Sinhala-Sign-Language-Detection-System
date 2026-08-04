import pandas as pd
import json

translations = {
    "Eat": "කන්න",
    "Good": "හොඳ",
    "Hello": "ආයුබෝවන්",
    "House": "ගෙදර",
    "Thank you": "ස්තුතියි"
}

csv_path = 'data/splits/sinhala_word_map.csv'
df = pd.read_csv(csv_path)

# Map the english name to sinhala
df['class_name_sinhala'] = df['class_name_english'].map(translations)

# If any are missing, keep english
df['class_name_sinhala'] = df['class_name_sinhala'].fillna(df['class_name_english'])

df.to_csv(csv_path, index=False)
print("Successfully translated 5 classes to Sinhala!")
