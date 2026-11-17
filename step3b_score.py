"""
Step 3b - Which recurring series are real DATA (panel-able), not just notices? (API, cheap)

Step 3 finds series that recur often, but many are routine notices or directories.
Here Claude scores each panel candidate from step 3, using the series name plus
6 example descriptions spread across decades, on:
  data_kind       : "statistical"   (counts, amounts, measurements - e.g. deaths, rainfall, arrests)
                    "itemized"      (one row per payment / property / person - can be aggregated)
                    "notice"        (notices, directories, appointments, schedules)
  unit            : what one observation is (e.g. "ward-week", "city-month", "payment")
  panel_potential : high / medium / low
  reason          : one short sentence
Answers are saved to data/step3b_raw.jsonl as they arrive (re-runs skip finished batches).

Output: output/panel_shortlist.csv  (candidates ranked: high-potential statistical series first)
"""
import json, os, re, time
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv(); client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"; BATCH = 20; RAW = "data/step3b_raw.jsonl"

summ = pd.read_csv("output/recurrence_summary.csv")
cand = summ[summ["panel_candidate"]].reset_index(drop=True)
objs = pd.read_pickle("data/objects_classified.pkl")
N_EX = 6   # examples per series, spread across decades so Claude sees the series over its whole life

def spread(g):
    g = g.dropna(subset=["Content"])
    if len(g) <= N_EX:
        return list(g.sort_values("year")["Content"])
    dec = g["year"] // 10 * 10
    per = max(1, N_EX // dec.nunique())
    pick = g.groupby(dec, group_keys=False).apply(lambda d: d.sample(min(per, len(d)), random_state=0))
    if len(pick) < N_EX:
        rest = g.drop(pick.index)
        pick = pd.concat([pick, rest.sample(min(N_EX - len(pick), len(rest)), random_state=0)])
    return list(pick.sample(min(N_EX, len(pick)), random_state=0).sort_values("year")["Content"])

ex = (objs[objs["series"].isin(cand["series"])]
      .groupby("series")[["Content", "year"]]
      .apply(spread))
cand["examples"] = cand["series"].map(ex)
cand["cid"] = cand.index

SYSTEM = """You assess series of tables/lists from the New York City Record (1890-1930) for a research project
that wants data which can become a PANEL or TIME SERIES dataset after digitization (recurring information
with a regular cadence, ideally numbers by place and time).
For each series return: cid, data_kind ("statistical" | "itemized" | "notice"), unit (what one observation is,
e.g. "ward-week", "borough-month", "payment", "property"), panel_potential ("high" | "medium" | "low"),
reason (one short sentence). Statistical summaries by place/time are usually high; itemized records that can be
aggregated are usually medium; notices, directories, appointments and schedules are usually low.
Return ONLY a JSON list."""

done = set()
if os.path.exists(RAW):
    done = {json.loads(l)["cid"] for l in open(RAW) if l.strip()}
todo = cand[~cand["cid"].isin(done)]
print(f"{len(done)} done, {len(todo)} to go")
for start in range(0, len(todo), BATCH):
    b = todo.iloc[start:start + BATCH]
    lines = [f"cid {r.cid}: {r.series} | geography: {r.geography} | ~{r.issues_per_year} issues/yr, {r.first_year}-{r.last_year}\n"
             + "\n".join(f"   - {e}" for e in r.examples) for r in b.itertuples()]
    for attempt in range(3):
        try:
            msg = client.messages.create(model=MODEL, max_tokens=8000, system=SYSTEM,
                                         messages=[{"role": "user", "content": "\n\n".join(lines)}])
            text = "".join(x.text for x in msg.content if getattr(x, "type", "") == "text")
            res = json.loads(re.search(r"\[.*\]", text, re.S).group(0)); break
        except Exception as e:
            print("  retrying after error:", e); time.sleep(5)
    else:
        print("  batch failed, skipping"); continue
    with open(RAW, "a") as f:
        for x in res: f.write(json.dumps(x) + "\n")
    print(f"  candidates {b.cid.min()}-{b.cid.max()} done")

sc = pd.read_json(RAW, lines=True).drop_duplicates("cid")
out = cand.drop(columns=["examples"]).merge(sc, on="cid", how="left")
order = {"high": 0, "medium": 1, "low": 2}
kind = {"statistical": 0, "itemized": 1, "notice": 2}
out = out.sort_values(by=["panel_potential", "data_kind", "n_issues"],
                      key=lambda c: c.map(order) if c.name == "panel_potential" else (c.map(kind) if c.name == "data_kind" else -c))
out.drop(columns=["cid"]).to_csv("output/panel_shortlist.csv", index=False)
print(out["panel_potential"].value_counts().to_dict(), out["data_kind"].value_counts().to_dict())
print(out[["series","data_kind","unit","panel_potential","cadence","geography","first_year","last_year"]].head(25).to_string(index=False))
