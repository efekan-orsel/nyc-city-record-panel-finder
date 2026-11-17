"""
Step 1 - Load the City Record database and read the date/page off every object.

Input : ../multimonth3.dta   (1.3M objects, one row per table/list on a page)
Output: data/objects.pkl     (same rows + clean date, page, object number)
"""
import pandas as pd

DTA_PATH = "../multimonth3.dta"

# 1) Read the Stata file in chunks (it is 1.5 GB, so we don't load it all at once)
chunks = []
with pd.read_stata(DTA_PATH, iterator=True) as reader:
    while True:
        try:
            chunk = reader.read(200_000)
        except StopIteration:
            break
        if chunk is None or len(chunk) == 0:
            break
        chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# 2) The ID looks like "3-1914-12-01-1" = object 3, issue of 1914-12-01, page 1.
#    Split it into separate columns.
parts = df["ID"].str.extract(r"^(\d+)-(\d{4})-(\d{2})-(\d{2})-(\d+)$")
df["object_no"] = parts[0].astype(int)
df["year"] = parts[1].astype(int)
df["month"] = parts[2].astype(int)
df["page"] = parts[4].astype(int)
df["issue"] = parts[1] + "-" + parts[2] + "-" + parts[3]   # issue label as printed in the ID

# Some IDs contain impossible dates (e.g. 1908-02-31, day "00"). Keep year/month,
# set the full date to missing, and flag them so we can report it.
df["date"] = pd.to_datetime(df["issue"], format="%Y-%m-%d", errors="coerce")
df["bad_date"] = df["date"].isna()

# 3) Save a fast-loading copy for the next steps
df.to_pickle("data/objects.pkl")

# 4) Quick sanity checks
print("Objects:", len(df))
print("Issues (distinct dates):", df["issue"].nunique())
print("Objects with impossible dates:", int(df["bad_date"].sum()))
print("Years:", df["year"].min(), "-", df["year"].max())
print("Type counts:\n", df["Type"].value_counts(dropna=False))
