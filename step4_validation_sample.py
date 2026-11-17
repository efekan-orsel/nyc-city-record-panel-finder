"""
Step 4 - Hand-check (validation). Free.

How much can we trust Claude's labels? We take the top 15 "high potential" series
from the shortlist and draw 5 random tables from each (75 in total). You open
output/validation_sample.csv in Excel and, for each row, fill in the column
"correct" with Y (the description really is that kind of table) or N (it is not).
Then run step4b_accuracy.py to get the accuracy.

(Optional, stronger check: open the actual scan for a few rows using the issue
date and page number.)
"""
import pandas as pd

short = pd.read_csv("output/panel_shortlist.csv")
top = short[short["panel_potential"] == "high"].head(15)["series"].tolist()

objs = pd.read_pickle("data/objects_classified.pkl")
sample = (objs[objs["series"].isin(top)]
          .groupby("series", group_keys=False)
          .apply(lambda s: s.sample(min(5, len(s)), random_state=1)))

sample = sample.merge(short[["series", "unit", "geography"]].rename(columns={"geography": "series_geography"}),
                      on="series", how="left")
cols = ["series", "unit", "series_geography", "ID", "issue", "page", "Type", "Content"]
out = sample[cols].copy()
out["correct"] = ""          # fill in Y / N
out["notes"] = ""            # optional comment
out.to_csv("output/validation_sample.csv", index=False)
print("Series checked:", len(top))
print("Rows to check:", len(out))
print("\n".join(f" - {s}" for s in top))
