import gradio as gr
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

JD_TEXT = """
Senior AI Engineer at Redrob AI. 5-9 years experience. 
Production experience with embeddings-based retrieval systems, vector databases, hybrid search.
Strong Python. Hands-on evaluation frameworks for ranking systems (NDCG, MRR, MAP).
Shipped end-to-end ranking, search, or recommendation system to real users at scale.
Product company experience preferred. NOT consulting firms like TCS, Infosys, Wipro, Accenture.
Location: Pune or Noida India preferred. Open to Hyderabad, Mumbai, Delhi NCR.
NOT pure researchers. Must write production code.
Scrappy product-engineering attitude. LLM fine-tuning, learning-to-rank experience a plus.
Active on platform, willing to relocate, short notice period preferred.
"""

CONSULTING_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant",
                    "capgemini", "mindtree", "mphasis", "hexaware", "tech mahindra"]

INDIA_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi",
                   "bangalore", "bengaluru", "chennai", "india"]

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
jd_embedding = model.encode([JD_TEXT])[0]

def build_candidate_text(c):
    p = c.get("profile", {})
    parts = []
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(f"Current role: {p.get('current_title','')} at {p.get('current_company','')}")
    parts.append(f"Experience: {p.get('years_of_experience', 0)} years")
    parts.append(f"Location: {p.get('location','')}, {p.get('country','')}")
    for job in c.get("career_history", [])[:3]:
        parts.append(job.get("description", ""))
    skills = [s["name"] for s in c.get("skills", [])]
    parts.append("Skills: " + ", ".join(skills))
    return " ".join(filter(None, parts))

def signal_score(c):
    s = c.get("redrob_signals", {})
    score = 0.0
    days = s.get("days_since_last_login", 999)
    if days <= 7:    score += 0.15
    elif days <= 30: score += 0.10
    elif days <= 90: score += 0.05
    else:            score -= 0.05
    score += s.get("recruiter_response_rate", 0) * 0.15
    score += s.get("profile_completeness_score", 0) * 0.10
    score += min(s.get("saved_by_recruiters_30d", 0) / 20, 1.0) * 0.05
    score += s.get("interview_completion_rate", 0) * 0.05
    return score

def rule_score(c):
    p = c.get("profile", {})
    score = 0.0
    yoe = p.get("years_of_experience", 0)
    if 5 <= yoe <= 9:    score += 0.20
    elif 4 <= yoe <= 11: score += 0.10
    elif yoe < 3:        score -= 0.15
    loc = (p.get("location", "") + p.get("country", "")).lower()
    if any(city in loc for city in INDIA_LOCATIONS):
        score += 0.15
    company = p.get("current_company", "").lower()
    all_companies = [j.get("company", "").lower() for j in c.get("career_history", [])]
    all_companies.append(company)
    consulting_count = sum(1 for co in all_companies if any(f in co for f in CONSULTING_FIRMS))
    if consulting_count >= 2: score -= 0.20
    elif consulting_count == 1: score -= 0.05
    title = p.get("current_title", "").lower()
    if any(t in title for t in ["ai", "ml", "machine learning", "nlp", "search", "ranking"]):
        score += 0.10
    return score

def generate_reasoning(c):
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    loc = f"{p.get('location','')}, {p.get('country','')}"
    skills = [s["name"] for s in c.get("skills", [])[:4]]
    signals = c.get("redrob_signals", {})
    days = signals.get("days_since_last_login", 999)
    rr = signals.get("recruiter_response_rate", 0)
    return f"{yoe}yr {title} at {company} ({loc}); skills: {', '.join(skills)}; last active {days}d ago, {int(rr*100)}% recruiter response rate."

def rank_candidates(file):
    if file is None:
        return "Please upload a JSONL file", None

    candidates = []
    with open(file.name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    if len(candidates) == 0:
        return "No candidates found in file", None

    texts = [build_candidate_text(c) for c in candidates]
    embeddings = model.encode(texts, show_progress_bar=False)
    semantic_scores = cosine_similarity([jd_embedding], embeddings)[0]

    final_scores = []
    for i, c in enumerate(candidates):
        sem = float(semantic_scores[i])
        sig = signal_score(c)
        rule = rule_score(c)
        final = (sem * 0.55) + (sig * 0.25) + (rule * 0.20)
        final_scores.append((c["candidate_id"], final, c))

    final_scores.sort(key=lambda x: (-x[1], x[0]))
    top = final_scores[:100] if len(final_scores) >= 100 else final_scores

    rows = []
    for rank, (cid, score, c) in enumerate(top, 1):
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 4),
            "reasoning": generate_reasoning(c)
        })

    df = pd.DataFrame(rows)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)

    preview = f"✅ Ranked {len(df)} candidates successfully!\n\nTop 5:\n"
    preview += df.head().to_string()

    return preview, tmp.name

with gr.Blocks(title="V4Vin — Intelligent Candidate Ranker") as demo:
    gr.Markdown("""
    # 🏆 V4Vin — Intelligent Candidate Ranking System
    ### India Runs Hackathon | Redrob AI | Track 1
    
    Upload a JSONL file containing candidate profiles and get them ranked for the **Senior AI Engineer** role.
    
    **Scoring:** 55% Semantic AI + 25% Behavioral Signals + 20% Rule-based Logic
    """)

    with gr.Row():
        file_input = gr.File(label="Upload candidates JSONL file", file_types=[".jsonl"])

    with gr.Row():
        run_btn = gr.Button("🚀 Rank Candidates", variant="primary")

    with gr.Row():
        output_text = gr.Textbox(label="Results", lines=15)
        output_file = gr.File(label="Download submission.csv")

    run_btn.click(fn=rank_candidates, inputs=[file_input], outputs=[output_text, output_file])

demo.launch()