import pandas as pd

df = pd.read_csv("submission.csv")

# Sort by score desc, then candidate_id asc for tie-breaking
df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True])

# Reassign ranks
df["rank"] = range(1, len(df) + 1)

df.to_csv("submission.csv", index=False)
print("✅ Tie-break fixed!")
print(df[["candidate_id", "rank", "score"]].head(20).to_string())