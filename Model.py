import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
import re
from nltk.corpus import stopwords


# === 1. Load Dataset ===
# Sample format: columns - 'review', 'rating'
df = pd.read_csv(r"C:\Users\garga\OneDrive\Desktop\Amazon review analysis\Amazon_Reviews_Cleaned.csv")  # Ensure this file has 'review' and 'rating' columns

# === 2. Label Sentiment ===
def label_sentiment(rating):
    if rating <= 2:
        return 'Bad'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Good'

df['sentiment'] = df['Rating'].apply(label_sentiment)

# === 3. Text Cleaning ===
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)        # Remove HTML tags
    text = re.sub(r'[^a-zA-Z ]', '', text)   # Remove punctuation/numbers
    text = text.lower().strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['cleaned_review'] = df['Review Text'].apply(clean_text)

# === 4. TF-IDF Vectorization ===
X = df['cleaned_review']
y = df['sentiment']

tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

# === 5. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y )

# === 6. Train Model ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === 7. Evaluate ===
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 8. Save Model and Vectorizer ===
import pickle
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
