from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.cache import cache_control
import os
import joblib
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from .models import Comment

# ---------------- NLTK SETUP ----------------
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'final_spam_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'final_vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- INDEX VIEW ----------------
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    if request.method == "POST":
        un = request.POST.get('username')
        up = request.POST.get('password')

        if un == "techwithvp" and up == "techwithvp":
            request.session['authdetails'] = "techwithvp"
        else:
            return render(request, 'auth.html', {"error": "Invalid credentials"})

    if request.session.get('authdetails') == "techwithvp":

        comments = Comment.objects.filter(result__icontains="Not Spam").order_by('-id')

        # 🔥 STATS (NEW)
        total = Comment.objects.count()
        spam_count = Comment.objects.filter(result__icontains="Spam").count()
        safe_count = Comment.objects.filter(result__icontains="Not Spam").count()

        return render(request, 'index.html', {
            "comments": comments,
            "total": total,
            "spam_count": spam_count,
            "safe_count": safe_count
        })

    return render(request, 'auth.html')

# ---------------- CHECK SPAM ----------------
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def checkSpam(request):
    if request.method == "POST" and request.session.get('authdetails') == "techwithvp":

        rawData = request.POST.get("rawdata")

        if not rawData:
            comments = Comment.objects.filter(result__icontains="Not Spam").order_by('-id')
            return render(request, 'index.html', {
                "comments": comments,
                "error": "Please enter text"
            })

        # CLEAN + VECTORIZE
        cleaned = clean_text(rawData)
        vector = vectorizer.transform([cleaned])

        # ML PREDICTION
        pred = model.predict(vector)[0]

        # 🔥 SAFE PROBABILITY (if available)
        try:
            prob = model.predict_proba(vector)[0][1]
            confidence = round(prob * 100, 2)
        except:
            confidence = "N/A (SVM)"

        # RULE-BASED BOOST
        text_lower = rawData.lower()
        spam_keywords = [
            "free", "win", "winner", "lottery", "click here",
            "urgent", "offer", "money", "cash", "prize", "claim"
        ]

        rule_score = sum(1 for word in spam_keywords if word in text_lower)

        # FINAL DECISION
        if rule_score >= 1 or pred == 1:
            result = "Spam"
        else:
            result = "Not Spam"

        # SAVE
        Comment.objects.create(text=rawData, result=result)

        # FETCH UPDATED DATA
        comments = Comment.objects.filter(result__icontains="Not Spam").order_by('-id')

        total = Comment.objects.count()
        spam_count = Comment.objects.filter(result__icontains="Spam").count()
        safe_count = Comment.objects.filter(result__icontains="Not Spam").count()

        return render(request, 'index.html', {
            "comments": comments,
            "last_result": result,
            "confidence": confidence,
            "total": total,
            "spam_count": spam_count,
            "safe_count": safe_count
        })

    return redirect('/')

# ---------------- OUTPUT PAGE ----------------
def output_page(request):
    if request.session.get('authdetails'):
        all_comments = Comment.objects.all().order_by('-id')
        return render(request, 'output.html', {
            "all_comments": all_comments
        })
    return redirect('/')

# ---------------- LOGOUT ----------------
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def logout(request):
    request.session.flush()
    return redirect('/')

# ---------------- API ----------------
def api_predict(request):
    text = request.GET.get('text')

    if not text:
        return JsonResponse({"error": "No input provided"})

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    pred = model.predict(vector)[0]

    text_lower = text.lower()
    spam_keywords = [
        "free", "win", "winner", "lottery", "click here",
        "urgent", "offer", "money", "cash", "prize", "claim"
    ]

    rule_score = sum(1 for word in spam_keywords if word in text_lower)

    result = "Spam" if rule_score >= 1 or pred == 1 else "Not Spam"

    return JsonResponse({
        "input": text,
        "prediction": result
    })