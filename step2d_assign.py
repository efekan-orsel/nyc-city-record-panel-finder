"""
Step 2d - Give EVERY object (all 1.3M rows) a clean series name. Free, no API.

For each label group:
  - if Claude said the group is coherent -> every object in it gets that series name
  - if the group is mixed -> Claude told us which of the 20 examples belong to which subtype.
    We compare each object's description to those examples (TF-IDF text similarity)
    and give it the subtype it is most similar to.
Objects in rare label groups (fewer than 50 issues) are labelled "Rare / unclassified".
Finally the names are replaced by the canonical names from step 2c.

Output: data/objects_classified.pkl
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

objs = pd.read_pickle("data/objects.pkl")
objs = objs[objs["Type"].isin(["Table", "List"])].copy()
objs["sub1"] = objs["Subtopic_1"].str.lower().str.strip()
objs["Content"] = objs["Content"].fillna("")

groups = pd.read_pickle("data/groups.pkl")
res = pd.read_json("data/step2b_raw_v2.jsonl", lines=True)
groups = groups.merge(res, on="group_id", how="left")
name_map = dict(pd.read_csv("data/name_map.csv").values)

objs = objs.merge(groups[["Topic", "sub1", "group_id"]], on=["Topic", "sub1"], how="left")
objs["series"] = "Rare / unclassified"
objs["geography"] = "unknown"
objs["match_sim"] = np.nan

for _, g in groups.iterrows():
    idx = objs.index[objs["group_id"] == g["group_id"]]
    if len(idx) == 0:
        continue
    subs = g["subtypes"] if isinstance(g["subtypes"], list) else []
    if g["coherent"] != "mixed" or len(subs) == 0:
        objs.loc[idx, "series"] = g["series_name"]
        objs.loc[idx, "geography"] = g["geography"]
        continue
    # Mixed group: build one "profile" per subtype from its example descriptions
    ex = g["examples"]
    texts = objs.loc[idx, "Content"].tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1).fit(texts + ex)
    X = vec.transform(texts)
    prof, labels, geos = [], [], []
    for s in subs:
        members = [ex[i - 1] for i in s.get("examples", []) if 1 <= i <= len(ex)]
        if members:
            prof.append(np.asarray(vec.transform(members).mean(axis=0)))
            labels.append(s["series_name"]); geos.append(s["geography"])
    P = np.vstack(prof)
    sim = cosine_similarity(X, P)
    best = sim.argmax(axis=1)
    objs.loc[idx, "series"] = [labels[b] for b in best]
    objs.loc[idx, "geography"] = [geos[b] for b in best]
    objs.loc[idx, "match_sim"] = sim.max(axis=1)

objs["series"] = objs["series"].map(lambda n: name_map.get(n, n))
objs.to_pickle("data/objects_classified.pkl")
print("Objects labelled:", len(objs))
print("Share in rare/unclassified:", round((objs.series == "Rare / unclassified").mean(), 3))
print("Distinct clean series:", objs["series"].nunique())
