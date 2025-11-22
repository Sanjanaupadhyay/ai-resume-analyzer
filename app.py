from flask import Flask, render_template, request, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import fitz   # PyMuPDF
import os
import tempfile
import heapq
import io

app = Flask(__name__)

# -----------------------
# Helpers
# -----------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9+\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(file_stream) -> str:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(file_stream.read())
            try:
                file_stream.seek(0)
            except Exception:
                pass

        doc = fitz.open(tmp_path)
        text_chunks = []
        for page in doc:
            text_chunks.append(page.get_text("text"))
        doc.close()
        return "\n".join(text_chunks)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def calculate_match(resume_text: str, jd_text: str) -> dict:
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    if not resume_clean or not jd_clean:
        return {
            "match_percent": 0,
            "missing_skills": [],
        }

    docs = [resume_clean, jd_clean]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    match_percent = round(similarity * 100, 2)

    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())

    jd_keywords = {w for w in jd_words if len(w) >= 3}
    missing_skills = sorted(list(jd_keywords - resume_words))
    missing_skills = missing_skills[:20]

    return {
        "match_percent": match_percent,
        "missing_skills": missing_skills,
    }

def calculate_ats_score(resume_text, jd_text, match_percent):
    ats_score = match_percent
    resume_lower = resume_text.lower()

    # Contact info check
    if ("@" in resume_text) and any(c.isdigit() for c in resume_text):
        ats_score += 5
    else:
        ats_score -= 5

    # Resume length check (ideal 150–500 words)
    word_count = len(resume_text.split())
    if 150 <= word_count <= 500:
        ats_score += 5
    elif word_count < 100:
        ats_score -= 5
    else:
        ats_score -= 2

    # Action verbs bonus
    action_verbs = [
        "developed", "built", "created", "analyzed", "designed",
        "managed", "optimized", "implemented", "led", "improved"
    ]
    if any(verb in resume_lower for verb in action_verbs):
        ats_score += 5

    # Skill keyword density
    jd_words = set(jd_text.lower().split())
    resume_words = resume_lower.split()
    overlap = [w for w in resume_words if w in jd_words]

    if len(jd_words) > 0:
        density = (len(overlap) / len(jd_words)) * 100
        if density > 30:
            ats_score += 5
        else:
            ats_score -= 5

    ats_score = max(0, min(100, ats_score))

    suggestions = []
    if "github" not in resume_lower:
        suggestions.append("Add your GitHub profile for more credibility.")
    if "project" not in resume_lower:
        suggestions.append("Include at least one technical project in detail.")
    if "sql" not in resume_lower and "database" not in resume_lower:
        suggestions.append("Mention your SQL or database experience if you have it.")
    if "python" not in resume_lower:
        suggestions.append("Include Python-related experience if applicable.")
    if len(suggestions) == 0:
        suggestions.append("Your resume structure looks good. Only minor improvements needed!")

    return int(ats_score), suggestions

def generate_summary(text: str, max_sentences: int = 3) -> str:
    if not text or len(text.split()) < 30:
        return text.strip()[:1000]

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r'\w+', text.lower())
    stopwords = set([
        'the','and','is','in','to','of','with','a','for','on','as','by','an','be','are','or','that','this',
        'it','from','at','we','our','you','your','have','has','will','can','using','used'
    ])
    freq = {}
    for w in words:
        if w in stopwords or len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1

    if not freq:
        return " ".join(sentences[:max_sentences])

    sent_scores = []
    for i, s in enumerate(sentences):
        s_words = re.findall(r'\w+', s.lower())
        score = 0
        for w in s_words:
            score += freq.get(w, 0)
        sent_scores.append((score, i, s))

    top = heapq.nlargest(max_sentences, sent_scores, key=lambda x: x[0])
    top_sorted = sorted(top, key=lambda x: x[1])
    summary = " ".join([t[2] for t in top_sorted])
    return summary.strip()

# -----------------------
# Routes
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    ats_score = 0
    suggestions = []
    resume_text = ""
    jd_text = ""
    summary = ""

    if request.method == "POST":
        # 1) try file upload first (if you implemented upload)
        uploaded_file = request.files.get("resume_file")
        if uploaded_file and uploaded_file.filename != "":
            filename = uploaded_file.filename.lower()
            if filename.endswith(".pdf"):
                try:
                    resume_text = extract_text_from_pdf(uploaded_file.stream)
                except Exception:
                    try:
                        uploaded_file.stream.seek(0)
                        resume_text = uploaded_file.stream.read().decode("utf-8", errors="ignore")
                    except Exception:
                        resume_text = ""
            else:
                try:
                    uploaded_file.stream.seek(0)
                    resume_text = uploaded_file.stream.read().decode("utf-8", errors="ignore")
                except Exception:
                    resume_text = ""

        # 2) fallback to textarea
        if not resume_text:
            resume_text = request.form.get("resume_text", "")

        jd_text = request.form.get("jd_text", "")

        # run match %
        result = calculate_match(resume_text, jd_text)

        # run ATS score
        ats_score, suggestions = calculate_ats_score(resume_text, jd_text, result["match_percent"])

        # generate summary
        summary = generate_summary(resume_text, max_sentences=3)

    return render_template(
        "index.html",
        result=result,
        resume_text=resume_text,
        jd_text=jd_text,
        ats_score=ats_score,
        suggestions=suggestions,
        summary=summary
    )


@app.route("/download_report", methods=["POST"])
def download_report():
    resume_text = request.form.get("resume_text", "")
    jd_text = request.form.get("jd_text", "")
    ats_score = request.form.get("ats_score", "")
    match_percent = request.form.get("match_percent", "")
    missing_skills = request.form.get("missing_skills", "")
    suggestions = request.form.get("suggestions", "")

    summary = generate_summary(resume_text, max_sentences=4)

    report_lines = [
        "AI Resume Analyzer - Report",
        "---------------------------",
        f"Match Percentage: {match_percent} %",
        f"ATS Score: {ats_score} / 100",
        "",
        "Missing Skills:",
        missing_skills,
        "",
        "Suggestions:",
        suggestions,
        "",
        "Resume Summary:",
        summary,
        "",
        "Job Description:",
        jd_text[:3000]
    ]

    report_text = "\n".join(report_lines)

    mem = io.BytesIO()
    mem.write(report_text.encode("utf-8"))
    mem.seek(0)

    return send_file(
        mem,
        as_attachment=True,
        download_name="resume_report.txt",
        mimetype="text/plain"
    )

import os

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
