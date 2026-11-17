"""
Step 4b - Compute accuracy from your hand-check in output/validation_sample.csv.
"""
import pandas as pd

v = pd.read_csv("output/validation_sample.csv")
v["correct"] = v["correct"].fillna("").astype(str).str.strip().str.upper().str[:1]   # blanks stay blank (not "N")
checked = v[v["correct"].isin(["Y", "N"])]
if checked.empty:
    raise SystemExit("No Y/N answers found yet - fill in the 'correct' column and save the CSV first.")
acc = (checked["correct"] == "Y").mean()
unsure = (v["correct"] == "U").sum()
print(f"Checked {len(checked)} of {len(v)} rows | overall accuracy: {acc:.1%} | marked unsure (U, excluded): {unsure}")
print("\nBy series:")
print(checked.groupby("series")["correct"].apply(lambda s: f"{(s == 'Y').mean():.0%} ({(s == 'Y').sum()}/{len(s)})").to_string())
