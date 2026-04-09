import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (run once)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("sms.tsv", sep="\t", header=None)
df.columns = ["label", "message"]

# Initialize tools
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords + stemming
    words = [ps.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply cleaning
df["cleaned"] = df["message"].apply(clean_text)

# Show before/after
for i in range(5):
    print("\nORIGINAL:", df["message"][i])
    print("CLEANED :", df["cleaned"][i])