import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("sms.tsv", sep="\t", header=None)
df.columns = ["label", "message"]

# Cleaning (same as before)
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

# Labels
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1,2),   # unigrams + bigrams
    max_features=5000
)
X = tfidf.fit_transform(df["cleaned"])
y = df["label_num"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Model 1: Naive Bayes --------
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# -------- Model 2: Logistic Regression --------
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

test_msg = ["URGENT! Click here now to win money"]
test_clean = [clean_text(msg) for msg in test_msg]
test_vec = tfidf.transform(test_clean)

print("Prediction:", lr.predict(test_vec))

from sklearn.svm import LinearSVC

svm = LinearSVC(class_weight='balanced')
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Report:\n", classification_report(y_test, y_pred_svm))


import joblib
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords (once)
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load final SVM + TF-IDF
model = joblib.load("final_spam_model.pkl")
vec = joblib.load("final_vectorizer.pkl")

# Cleaning function (must match training exactly)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Prediction function
def predict_spam(text):
    cleaned = [clean_text(text)]
    vector = vec.transform(cleaned)
    result = model.predict(vector)[0]
    return "SPAM" if result == 1 else "NOT SPAM"

# TEST
print(predict_spam("URGENT! Click here now to win money"))
print(predict_spam("Hey, are we meeting today?"))