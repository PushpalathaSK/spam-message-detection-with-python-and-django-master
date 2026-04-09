import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# LOAD DATA
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# MAP LABELS
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔥 ADD CUSTOM DATA
extra_data = pd.DataFrame({
    'label': [1,1,1,1,0,0,0],
    'message': [
        "urgent click here now",
        "limited time offer",
        "free money now",
        "claim your prize",
        "let's meet tomorrow",
        "call me later",
        "are you coming"
    ]
})

df = pd.concat([df, extra_data], ignore_index=True)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# MODEL
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,2),
        max_features=5000,
        stop_words='english'
    )),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ))
])

# TRAIN
model.fit(X_train, y_train)

# EVALUATE
print("Accuracy:", model.score(X_test, y_test))

# SAVE
joblib.dump(model, "final_spam_model.pkl")
print("Model saved!")