"""
Step 2a - Group the 1.3M objects into candidate "label groups" (free, no API).

Idea: many objects share the same Topic + main Subtopic. We group by those,
count how many different issues each group appears in, and keep only groups that
show up often enough to possibly be a recurring (panel-able) table.
For each kept group we save 24 example descriptions (spread across decades) - those examples are what we
will later send to Claude (step 2b) to give the group a clean, standard name.
"""
import pandas as pd

MIN_ISSUES = 50          # a group must appear in at least this many issues

df = pd.read_pickle("data/objects.pkl")
df = df[df["Type"].isin(["Table", "List"])].copy()

# Light cleaning of the label: lower-case, trim spaces
df["sub1"] = df["Subtopic_1"].str.lower().str.strip()

groups = (
    df.groupby(["Topic", "sub1"])
      .agg(n_objects=("ID", "size"),
           n_issues=("issue", "nunique"),
           first_year=("year", "min"),
           last_year=("year", "max"),
           share_table=("Type", lambda s: (s == "Table").mean()))
      .reset_index()
)
keep = groups[groups["n_issues"] >= MIN_ISSUES].copy()

# 24 example descriptions per kept group, SPREAD ACROSS DECADES (fixed seed = reproducible).
# Purely random examples would mostly come from the decades where a table is most common;
# spreading them out lets Claude see how the table looks in every era it appears in.
N_EXAMPLES = 24

def spread_sample(g):
    g = g.dropna(subset=["Content"])
    if len(g) <= N_EXAMPLES:
        return list(g["Content"])
    decades = g["year"] // 10 * 10
    per_decade = max(1, N_EXAMPLES // decades.nunique())       # equal share per decade
    picked = (g.groupby(decades, group_keys=False)
               .apply(lambda d: d.sample(min(per_decade, len(d)), random_state=0)))
    if len(picked) < N_EXAMPLES:                               # top up randomly if a decade was small
        rest = g.drop(picked.index)
        picked = pd.concat([picked, rest.sample(min(N_EXAMPLES - len(picked), len(rest)), random_state=0)])
    picked = picked.head(N_EXAMPLES).sort_values("year")       # keep at most 24, in time order
    return list(picked["Content"])

examples = (
    df.merge(keep[["Topic", "sub1"]], on=["Topic", "sub1"])
      .groupby(["Topic", "sub1"])[["Content", "year"]]
      .apply(spread_sample)
      .rename("examples")
      .reset_index()
)
keep = keep.merge(examples, on=["Topic", "sub1"])
keep = keep.sort_values("n_issues", ascending=False).reset_index(drop=True)
keep["group_id"] = keep.index

keep.to_pickle("data/groups.pkl")
print("All label groups:", len(groups))
print(f"Groups in >= {MIN_ISSUES} issues:", len(keep))
print("Objects covered by kept groups:", int(keep["n_objects"].sum()), "of", len(df))
