"""
Step 2c - Merge duplicate series names (one API call).

Claude named each group in separate batches, so the same kind of table can have
slightly different names ("Municipal Court district directory" vs "Municipal court
district directory", "City payment vouchers" vs "Payment voucher lists").

1) Free pre-step in Python: names that are identical apart from capital letters /
   extra spaces are merged automatically.
2) We send Claude the remaining list ONCE and ask it to write, line by line,
   "number | canonical name". (A simple line format is much more robust than one
   giant JSON answer.)
3) Claude's raw answer is saved to data/step2c_raw.txt BEFORE we read it, so if
   reading fails we can fix the code and re-use the saved answer without paying again.

Output: data/name_map.csv  (name -> canonical_name)
"""
import os, re
import pandas as pd
from dotenv import load_dotenv
import anthropic

RAW_TXT = "data/step2c_raw.txt"
MODEL = "claude-opus-5-5"

# --- collect every name from step 2b (top-level + subtypes) ---
r = pd.read_json("data/step2b_raw_v2.jsonl", lines=True)
all_names = set(r["series_name"])
for subs in r["subtypes"]:
    for s in (subs or []):
        all_names.add(s["series_name"])

# --- 1) free merge: same name apart from case / spacing ---
def norm(n): return re.sub(r"\s+", " ", n.strip().lower())
first_seen = {}
for n in sorted(all_names):
    first_seen.setdefault(norm(n), n)
names = sorted(first_seen.values())            # one representative per normalised name
print("Distinct names from step 2b:", len(all_names), "| after free merge:", len(names))

# --- 2) ask Claude once (or re-use the saved answer) ---
if os.path.exists(RAW_TXT):
    text = open(RAW_TXT, encoding="utf-8").read()
    print("Re-using saved Claude answer from", RAW_TXT)
else:
    load_dotenv()
    client = anthropic.Anthropic()
    numbered = "\n".join(f"{i}: {n}" for i, n in enumerate(names))
    prompt = f"""Below is a numbered, alphabetical list of names for kinds of tables/lists printed in the
New York City Record (1890-1930). Several names describe the SAME kind of recurring table in different words.

For EVERY number, write one line in exactly this format:
number | canonical name

Give names that clearly describe the same kind of table (same content, same purpose) the SAME canonical name,
choosing the clearest short name. Be conservative: keep genuinely different tables apart.
Output only these lines, one per number, nothing else.

Names:
{numbered}"""
    with client.messages.stream(model=MODEL, max_tokens=32000,
                                messages=[{"role": "user", "content": prompt}]) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    open(RAW_TXT, "w", encoding="utf-8").write(text)
    print("Saved Claude's answer | tokens in:", msg.usage.input_tokens, "out:", msg.usage.output_tokens)

# --- 3) read the "number | canonical" lines (bad lines are just skipped) ---
canon_of = {}
for line in text.splitlines():
    m = re.match(r"\s*(\d+)\s*\|\s*(.+?)\s*$", line)
    if m and int(m.group(1)) < len(names):
        canon_of[int(m.group(1))] = m.group(2)

rows = []
for n in sorted(all_names):
    rep = first_seen[norm(n)]
    i = names.index(rep)
    rows.append({"name": n, "canonical_name": canon_of.get(i, rep)})   # missing -> keep own name
m = pd.DataFrame(rows)
m.to_csv("data/name_map.csv", index=False)
print("Names Claude mapped:", len(canon_of), "of", len(names))
print("Canonical series after merging:", m["canonical_name"].nunique())
