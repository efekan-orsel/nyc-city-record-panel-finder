"""
Step 5b - Request list for a transit-shock (bridges / subway) before-after design. Free.

Window 1878-1913 brackets: Brooklyn Bridge (1883; note Brooklyn itself is only covered
after the 1898 consolidation), Williamsburg Bridge (1903), IRT subway (1904),
subway to Brooklyn (1908), Manhattan and Queensboro Bridges (1909).
For each theme we pick, per year, the issue (closest to mid-year) whose descriptions
match the theme, and list its pages.

Themes are matched on the object descriptions (keyword patterns below).

Output: output/transit_request_list.csv, output/transit_request_issues.csv
"""
import pandas as pd

THEMES = {
    "Bridge traffic / tolls":            r"bridge.*(?:toll|traffic|passeng|vehicle|crossing)|(?:toll|traffic|passeng).*bridge",
    "Subway / rapid transit traffic":    r"(?:subway|rapid transit|interborough).*(?:passeng|earnings|receipts|operat|traffic)",
    "Deaths & disease by ward":          r"(?:deaths|disease|sickness).*ward|ward.*(?:deaths|disease|sickness)",
    "Vital statistics / population":     r"vital statistic|population|census",
    "Building plans filed / new buildings": r"new building|plans filed",
}

START, END = 1878, 1913      # study window (years)

o = pd.read_pickle("data/objects_classified.pkl")
o = o[~o["bad_date"] & o["year"].between(START, END)].copy()
text = o["Content"].str.lower()

rows = []
for theme, pat in THEMES.items():
    m = o[text.str.contains(pat, regex=True)]
    per_issue = m.groupby("issue").agg(date=("date", "first"), pages=("page", lambda p: sorted(set(p))))
    per_issue["year"] = per_issue["date"].dt.year
    for year in range(START, END + 1):
        yr = per_issue[per_issue["year"] == year]
        if yr.empty:
            continue
        mid = pd.Timestamp(f"{year}-07-01")
        best = yr.iloc[(yr["date"] - mid).abs().argsort()[:1]]
        rows.append({"theme": theme, "year": year, "issue": best.index[0],
                     "pages": ", ".join(map(str, best["pages"].iat[0]))})

req = pd.DataFrame(rows)
req.to_csv("output/transit_request_list.csv", index=False)
issues = sorted(req["issue"].unique())
pd.Series(issues, name="issue_date").to_csv("output/transit_request_issues.csv", index=False)
print("Rows (theme x year):", len(req), "| unique issues:", len(issues))
print(req.groupby("theme")["year"].agg(["count", "min", "max"]).to_string())
