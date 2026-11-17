"""
Step 5 - Build the list of City Record issues to request for a closer look. Free.

For each series that scored 100% in the hand-check (step 4) and is "high" panel potential,
pick ONE issue roughly every 5 years (closest to 1875, 1880, ..., 1945) where the series
appears, and list the page numbers. This lets us check on the real scans that each table
is what we think it is and whether its format changes over time.

Output: output/pdf_request_list.csv  (series, target_year, issue date, pages)
        output/pdf_request_issues.csv (unique issue dates to send to the team)
"""
import pandas as pd

v = pd.read_csv("output/validation_sample.csv")
v["correct"] = v["correct"].fillna("").astype(str).str.strip().str.upper().str[:1]
acc = v[v["correct"].isin(["Y", "N"])].groupby("series")["correct"].apply(lambda s: (s == "Y").mean())
reliable = acc[acc == 1.0].index.tolist()

FIRST_TARGET, LAST_TARGET = 1875, 1945   # target years, every 5 years

objs = pd.read_pickle("data/objects_classified.pkl")
objs = objs[objs["series"].isin(reliable) & ~objs["bad_date"]]

rows = []
for series, s in objs.groupby("series"):
    issue_pages = s.groupby("issue").agg(date=("date", "first"), pages=("page", lambda p: sorted(set(p))))
    issue_pages["year"] = issue_pages["date"].dt.year
    for target in range(FIRST_TARGET, LAST_TARGET + 1, 5):
        cand = issue_pages.iloc[(issue_pages["year"] - target).abs().argsort()[:1]]
        if len(cand) and abs(int(cand["year"].iat[0]) - target) <= 2:
            rows.append({"series": series, "target_year": target, "issue": cand.index[0],
                         "pages": ", ".join(map(str, cand["pages"].iat[0]))})
req = pd.DataFrame(rows).drop_duplicates(["series", "issue"])
req.to_csv("output/pdf_request_list.csv", index=False)
issues = sorted(req["issue"].unique())
pd.Series(issues, name="issue_date").to_csv("output/pdf_request_issues.csv", index=False)
print("Reliable series used:", len(reliable))
print("\n".join(" - " + r for r in reliable))
print("\nRows (series x year):", len(req), "| unique issues to request:", len(issues))
print(req.head(12).to_string(index=False))
