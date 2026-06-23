import pandas as pd
import json

translations = {
    "2. two": "දෙක",
    "Again": "නැවත",
    "Beautiful": "ලස්සන",
    "Cat": "බළලා",
    "Come": "එන්න",
    "Cook": "උයන්න",
    "Cry": "අඬන්න",
    "Cut": "කපන්න",
    "Day": "දවස",
    "Drink": "බොන්න",
    "Eat": "කන්න",
    "Elder bro": "අයියා",
    "Elder sister": "අක්කා",
    "Elephant": "අලියා",
    "Fight": "රණ්ඩු",
    "Go": "යන්න",
    "Good": "හොඳ",
    "Grand father": "සීයා",
    "Green": "කොළ",
    "Hello": "ආයුබෝවන්",
    "Help": "උදව්",
    "House": "ගෙදර",
    "I": "මම",
    "Look": "බලන්න",
    "Man": "මිනිසා",
    "Meet": "හමුවෙනවා",
    "Monday": "සඳුදා",
    "Money": "සල්ලි",
    "Mother": "අම්මා",
    "My": "මගේ",
    "Play": "සෙල්ලම්",
    "Purple": "දම්",
    "Red": "රතු",
    "Run": "දුවන්න",
    "See": "දකිනවා",
    "Sell": "විකුණනවා",
    "Sleep": "නිදාගන්නවා",
    "Squirrel": "ලේනා",
    "Teach": "උගන්වනවා",
    "Tell": "කියන්න",
    "Thank you": "ස්තුතියි",
    "Today": "අද",
    "Us": "අපිට",
    "Walk": "ඇවිදිනවා",
    "Want": "ඕන",
    "White": "සුදු",
    "Write": "ලියන්න",
    "Yellow": "කහ",
    "You": "ඔයා",
    "Younger bro": "මල්ලි"
}

csv_path = 'data/splits/sinhala_word_map.csv'
df = pd.read_csv(csv_path)

# Map the english name to sinhala
df['class_name_sinhala'] = df['class_name_english'].map(translations)

# If any are missing, keep english
df['class_name_sinhala'] = df['class_name_sinhala'].fillna(df['class_name_english'])

df.to_csv(csv_path, index=False)
print("Successfully translated 50 classes to Sinhala!")
