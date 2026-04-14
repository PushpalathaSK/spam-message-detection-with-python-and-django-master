from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.cache import cache_control
import os
import joblib
import re
import nltk
import time
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

    # LOGIN HANDLING
    if request.method == "POST":
        un = request.POST.get('username')
        up = request.POST.get('password')

        if un == "techwithvp" and up == "techwithvp":
            request.session['authdetails'] = "techwithvp"
        else:
            return render(request, 'auth.html', {"error": "Invalid credentials"})

    if request.session.get('authdetails') == "techwithvp":

        comments = Comment.objects.filter(result="Not Spam").order_by('-id')

        # 🔥 ANALYTICS
        total = Comment.objects.count()
        spam_count = Comment.objects.filter(result="Spam").count()
        safe_count = Comment.objects.filter(result="Not Spam").count()

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
            return redirect('/')

        start_time = time.time()

        # CLEAN + VECTORIZE
        cleaned = clean_text(rawData)
        vector = vectorizer.transform([cleaned])

        # ML PREDICTION
        pred = model.predict(vector)[0]

        # 🔥 CONFIDENCE
        try:
            prob = model.predict_proba(vector)[0][1]
            confidence = round(prob * 100, 2)
        except:
            try:
                score = model.decision_function(vector)[0]
                confidence = round(abs(score) * 10, 2)
            except:
                confidence = 0.0

        # RULE-BASED BOOST
        text_lower = rawData.lower()
        spam_keywords = [
            "free", "win", "winner", "lottery",
            "click here", "urgent", "offer",
            "money", "cash", "prize", "claim"
        ]

        rule_score = sum(1 for word in spam_keywords if word in text_lower)

        # FINAL DECISION
        if rule_score >= 1 or pred == 1:
            result = "Spam"
        else:
            result = "Not Spam"

        processing_time = round(time.time() - start_time, 3)

        # SAVE WITH CONFIDENCE
        Comment.objects.create(
            text=rawData,
            result=result,
            confidence=confidence
        )

        # REFRESH DATA
        comments = Comment.objects.filter(result="Not Spam").order_by('-id')

        total = Comment.objects.count()
        spam_count = Comment.objects.filter(result="Spam").count()
        safe_count = Comment.objects.filter(result="Not Spam").count()

        return render(request, 'index.html', {
            "comments": comments,
            "last_result": result,
            "confidence": confidence,
            "time": processing_time,
            "total": total,
            "spam_count": spam_count,
            "safe_count": safe_count
        })

    return redirect('/')

# ---------------- INSIGHTS PAGE ----------------
def insights(request):
    if request.session.get('authdetails') == "techwithvp":

        all_messages = Comment.objects.all().order_by('-id')

        total = all_messages.count()
        spam_count = all_messages.filter(result__icontains="Spam").count()
        safe_count = all_messages.filter(result__icontains="Not Spam").count()

        # 🔥 FEEDBACK MISTAKES
        mistakes = all_messages.exclude(feedback__isnull=True).count()

        # 🔥 ACCURACY CALCULATION (STEP 4)
        if total > 0:
            accuracy = round(((total - mistakes) / total) * 100, 2)
        else:
            accuracy = 0

        return render(request, 'insights.html', {
            "all_messages": all_messages,
            "total": total,
            "spam_count": spam_count,
            "safe_count": safe_count,
            "mistakes": mistakes,
            "accuracy": accuracy
        })

    return redirect('/')

# ---------------- FEEDBACK ----------------
def feedback(request, id, action):
    comment = Comment.objects.get(id=id)

    if action == "spam":
        comment.feedback = "Marked as Spam"
        comment.result = "Spam"   # 🔥 UPDATE RESULT
    elif action == "safe":
        comment.feedback = "Marked as Safe"
        comment.result = "Not Spam"   # 🔥 UPDATE RESULT

    comment.save()
    return redirect('/insights/')

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
        "free", "win", "winner", "lottery",
        "click here", "urgent", "offer",
        "money", "cash", "prize", "claim"
    ]

    rule_score = sum(1 for word in spam_keywords if word in text_lower)

    result = "Spam" if rule_score >= 1 or pred == 1 else "Not Spam"

    return JsonResponse({
        "input": text,
        "prediction": result
    })