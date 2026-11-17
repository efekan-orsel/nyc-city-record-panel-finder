"""
Step 3 - Which series recur regularly over time? (free)

For every clean series we measure:
  n_issues      : number of City Record issues it appears in
  first/last    : first and last year seen
  years_active  : number of distinct years it appears in
  coverage      : share of all issues in its active span that contain it
  issues_per_year: issues per active year containing it
  cadence       : weekly / monthly / quarterly / annual / irregular (from issues_per_year)
  geography     : most common geographic level
A series is flagged "panel candidate" if it appears in >= 100 issues across >= 10 years.

Output: output/recurrence_summary.csv  (sorted, best panel candidates first)
"""
import pandas as pd

o = pd.read_pickle("data/objects_classified.pkl")
o = o[o["series"] != "Rare / unclassified"]
issues_by_year = pd.read_pickle("data/objects.pkl").groupby("year")["issue"].nunique()

def cadence(per_year):
    # The City Record was published almost daily, so we judge cadence by how many
    # issues per active year contain the series (not by the gap between issues).
    if per_year >= 40: return "weekly or more often"
    if per_year >= 9:  return "monthly"
    if per_year >= 3:  return "quarterly"
    if per_year >= 0.8: return "annual"
    return "irregular"

rows = []
for name, s in o.groupby("series"):
    y0, y1 = s["year"].min(), s["year"].max()
    span_issues = issues_by_year.loc[y0:y1].sum()
    rows.append({
        "series": name,
        "n_objects": len(s),
        "n_issues": s["issue"].nunique(),
        "first_year": y0, "last_year": y1,
        "years_active": s["year"].nunique(),
        "coverage": round(s["issue"].nunique() / span_issues, 3),
        "issues_per_year": round(s["issue"].nunique() / s["year"].nunique(), 1),
        "cadence": cadence(s["issue"].nunique() / s["year"].nunique()),
        "geography": s["geography"].mode().iat[0],
        "share_tables": round((s["Type"] == "Table").mean(), 2),
    })
r = pd.DataFrame(rows)
r["panel_candidate"] = (r["n_issues"] >= 100) & (r["years_active"] >= 10)
r = r.sort_values(["panel_candidate", "n_issues"], ascending=False)
r.to_csv("output/recurrence_summary.csv", index=False)
print("Series:", len(r), "| panel candidates:", int(r["panel_candidate"].sum()))
print(r.head(15).to_string(index=False))
