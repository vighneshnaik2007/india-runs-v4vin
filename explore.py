import json

print("Loading sample candidates...")
with open("sample_candidates.json", "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Total sample candidates: {len(candidates)}")
print(f"\nKeys in each candidate: {list(candidates[0].keys())}")
print(f"\nKeys in 'profile': {list(candidates[0]['profile'].keys())}")
print(f"\nKeys in 'redrob_signals': {list(candidates[0]['redrob_signals'].keys())}")
print(f"\n--- Sample Candidate 1 ---")
print(f"Name: {candidates[0]['profile']['anonymized_name']}")
print(f"Headline: {candidates[0]['profile']['headline']}")
print(f"Location: {candidates[0]['profile']['location']}, {candidates[0]['profile']['country']}")
print(f"Experience: {candidates[0]['profile']['years_of_experience']} years")
print(f"Current role: {candidates[0]['profile']['current_title']} at {candidates[0]['profile']['current_company']}")