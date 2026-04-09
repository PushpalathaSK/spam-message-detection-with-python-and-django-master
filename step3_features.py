import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load dataset
df = pd.read_csv("sms.tsv", sep="\t", header=None)
df.columns = ["label", "message"]

# Load cleaned data (reuse cleaning logic OR save from previous step)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)

# Convert labels to numbers
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# -------- CountVectorizer --------
cv = CountVectorizer()
X_cv = cv.fit_transform(df["cleaned"])

print("CountVectorizer Shape:", X_cv.shape)

# -------- TF-IDF --------
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(df["cleaned"])

print("TF-IDF Shape:", X_tfidf.shape)