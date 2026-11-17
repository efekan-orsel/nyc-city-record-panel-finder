"""
Step 2b - Ask Claude (via the API) to give each label group a clean, standard name.

For each of the ~500 groups from step 2a we send Claude 24 example descriptions (spread across decades, in time order).
Claude returns, for each group:
  - series_name : short standard name for this kind of table (e.g. "Health Dept weekly mortality")
  - geography   : finest geographic level (address / ward-district-precinct / borough / city / none / mixed)
  - coherent    : "yes" if the 24 examples are the same kind of table, "mixed" if not
  - panel_note  : one short sentence on whether it could become a panel/time series

Groups are sent in batches of 10 to keep calls small. Every answer is saved to
data/step2b_raw.jsonl as soon as it arrives, so if the script stops halfway we can
re-run it and it skips batches that are already done.
"""
import json, os, re, time
import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()                                   # reads ANTHROPIC_API_KEY from the .env file
client = anthropic.Anthropic()                  # the key is picked up automatically
MODEL = os.getenv("CR_MODEL", "claude-opus-5-5")
BATCH = 10
RAW = "data/step2b_raw_v2.jsonl"   # 24 examples per group, spread across decades

SYSTEM = """You help classify objects (tables and lists) printed in the New York City Record, 1890-1930.
Each group below was formed from the database's existing Topic/Subtopic labels, which are noisy.
You see 24 numbered example descriptions per group, spread across the decades it appears in and listed in time order. Your job is to give each group a clean, STANDARD name
describing what kind of recurring table/list it is, so groups that are really the same kind of
object end up with the same name.

Rules:
- series_name: 2-7 words, generic and reusable (e.g. "City payment vouchers", "Health Dept mortality statistics",
  "Police precinct arrests", "Street improvement assessments"). Do not include specific years, people or streets.
- geography: the finest geographic unit the tables are broken down by:
  one of "address", "ward/district/precinct", "borough", "city", "none", "mixed".
- coherent: "yes" if at least ~80% of the 24 examples describe the same kind of object, otherwise "mixed".
- subtypes: ONLY if coherent is "mixed": a list of the distinct kinds of object in the group, each as
  {"series_name": ..., "geography": ..., "examples": [list of example numbers that belong to it]}.
  In that case the top-level series_name/geography describe the largest subtype. If coherent is "yes", use [].
- panel_note: one short sentence: could this become a regular panel / time series? Why or why not?
Base everything ONLY on the example descriptions. If unsure, say so in panel_note.

Return ONLY a JSON list, one object per group, with keys:
group_id, series_name, geography, coherent, subtypes, panel_note"""

def done_ids():
    if not os.path.exists(RAW):
        return set()
    with open(RAW) as f:
        return {json.loads(line)["group_id"] for line in f if line.strip()}

def ask_claude(batch_df):
    lines = []
    for _, r in batch_df.iterrows():
        ex = "\n".join(f"   {i+1}. {e}" for i, e in enumerate(r["examples"]))
        lines.append(f"group_id {r['group_id']} (label: {r['Topic']} / {r['sub1']}):\n{ex}")
    msg = client.messages.create(
        model=MODEL, max_tokens=8000, system=SYSTEM,
        messages=[{"role": "user", "content": "\n\n".join(lines)}],
    )
    # The reply can contain several blocks (e.g. a "thinking" block first); keep only the text blocks
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    text = re.search(r"\[.*\]", text, re.S).group(0)    # keep just the JSON list
    return json.loads(text), msg.usage

groups = pd.read_pickle("data/groups.pkl")
already = done_ids()
todo = groups[~groups["group_id"].isin(already)]
print(f"{len(already)} groups already done, {len(todo)} to go")

in_tok = out_tok = 0
for start in range(0, len(todo), BATCH):
    batch = todo.iloc[start:start + BATCH]
    for attempt in range(3):
        try:
            results, usage = ask_claude(batch)
            break
        except Exception as e:
            print("  retrying after error:", e)
            time.sleep(5)
    else:
        print("  batch failed 3 times, skipping for now"); continue
    in_tok += usage.input_tokens; out_tok += usage.output_tokens
    with open(RAW, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  groups {batch.group_id.min()}-{batch.group_id.max()} done")

print("Tokens used this run - input:", in_tok, "output:", out_tok)
