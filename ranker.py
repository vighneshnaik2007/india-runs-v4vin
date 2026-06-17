import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

start = time.time()

# ── 1. JOB DESCRIPTION TEXT ──────────────────────────────────────────────
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

# ── 2. LOAD MODEL ─────────────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
jd_embedding = model.encode([JD_TEXT])[0]

# ── 3. CONSULTING FIRMS TO PENALIZE ───────────────────────────────────────
CONSULTING_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant", 
                    "capgemini", "mindtree", "mphasis", "hexaware", "tech mahindra"]

INDIA_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi", 
                   "bangalore", "bengaluru", "chennai", "india"]

# ── 4. BUILD CANDIDATE TEXT ───────────────────────────────────────────────
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

# ── 5. SIGNAL SCORE ───────────────────────────────────────────────────────
def signal_score(c):
    s = c.get("redrob_signals", {})
    score = 0.0

    # Login recency (higher = more recent)
    days = s.get("days_since_last_login", 999)
    if days <= 7:    score += 0.15
    elif days <= 30: score += 0.10
    elif days <= 90: score += 0.05
    else:            score -= 0.05

    # Recruiter response rate
    rr = s.get("recruiter_response_rate", 0)
    score += rr * 0.15

    # Profile completeness
    score += s.get("profile_completeness_score", 0) * 0.10

    # Saved by recruiters
    saved = s.get("saved_by_recruiters_30d", 0)
    score += min(saved / 20, 1.0) * 0.05

    # Interview completion
    score += s.get("interview_completion_rate", 0) * 0.05

    return score

# ── 6. RULE-BASED SCORE ───────────────────────────────────────────────────
def rule_score(c):
    p = c.get("profile", {})
    score = 0.0

    # Experience range 5-9 years
    yoe = p.get("years_of_experience", 0)
    if 5 <= yoe <= 9:    score += 0.20
    elif 4 <= yoe <= 11: score += 0.10
    elif yoe < 3:        score -= 0.15

    # Location bonus
    loc = (p.get("location", "") + p.get("country", "")).lower()
    if any(city in loc for city in INDIA_LOCATIONS):
        score += 0.15

    # Consulting firm penalty
    company = p.get("current_company", "").lower()
    all_companies = [j.get("company", "").lower() for j in c.get("career_history", [])]
    all_companies.append(company)
    consulting_count = sum(1 for co in all_companies if any(f in co for f in CONSULTING_FIRMS))
    if consulting_count >= 2: score -= 0.20
    elif consulting_count == 1: score -= 0.05

    # Current title relevance
    title = p.get("current_title", "").lower()
    if any(t in title for t in ["ai", "ml", "machine learning", "nlp", "search", "ranking"]):
        score += 0.10

    return score

# ── 7. LOAD & SCORE ALL CANDIDATES ───────────────────────────────────────
print("Loading candidates.jsonl (this may take a minute)...")
candidates = []
with open("candidates.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f):
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

print(f"Loaded {len(candidates)} candidates")

# ── 8. EMBED ALL CANDIDATES ───────────────────────────────────────────────
print("Building candidate texts...")
texts = [build_candidate_text(c) for c in tqdm(candidates)]

print("Generating embeddings (this will take several minutes)...")
BATCH = 512
embeddings = []
for i in tqdm(range(0, len(texts), BATCH)):
    batch_emb = model.encode(texts[i:i+BATCH], show_progress_bar=False)
    embeddings.append(batch_emb)

embeddings = np.vstack(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings for reuse
np.save("embeddings.npy", embeddings)
print("Embeddings saved to embeddings.npy")

# ── 9. COMPUTE FINAL SCORES ───────────────────────────────────────────────
print("Computing scores...")
semantic_scores = cosine_similarity([jd_embedding], embeddings)[0]

final_scores = []
for i, c in enumerate(tqdm(candidates)):
    sem = float(semantic_scores[i])
    sig = signal_score(c)
    rule = rule_score(c)
    final = (sem * 0.55) + (sig * 0.25) + (rule * 0.20)
    final_scores.append((c["candidate_id"], final, c))

# ── 10. RANK TOP 100 ──────────────────────────────────────────────────────
final_scores.sort(key=lambda x: (-x[1], x[0]))
top100 = final_scores[:100]

# ── 11. GENERATE REASONING ────────────────────────────────────────────────
def generate_reasoning(c, score):
    p = c.get("profile", {})
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    loc = f"{p.get('location','')}, {p.get('country','')}"
    skills = [s["name"] for s in c.get("skills", [])[:4]]
    signals = c.get("redrob_signals", {})
    days = signals.get("days_since_last_login", 999)
    rr = signals.get("recruiter_response_rate", 0)
    
    reasoning = f"{yoe}yr {title} at {company} ({loc}); skills: {', '.join(skills)}; "
    reasoning += f"last active {days}d ago, {int(rr*100)}% recruiter response rate."
    return reasoning

# ── 12. WRITE CSV ─────────────────────────────────────────────────────────
rows = []
for rank, (cid, score, c) in enumerate(top100, 1):
    rows.append({
        "candidate_id": cid,
        "rank": rank,
        "score": round(score, 4),
        "reasoning": generate_reasoning(c, score)
    })

df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print(f"\n✅ submission.csv generated with {len(df)} candidates")
print(f"⏱ Total time: {round(time.time()-start, 1)}s")
print("\nTop 5 candidates:")
print(df.head().to_string())