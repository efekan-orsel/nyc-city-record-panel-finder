# Finding "panel-able" tables in the NYC City Record (1873–1947)

**Question:** Which tables in the City Record recur on a regular cadence over many years, so that
they could become a panel or time-series dataset after digitization?

**Data:** An object-level database of the City Record (≈1.86 million tables and lists from 22,110
issues, 1873–1947), where each object has an ID (object–date–page), noisy Topic/Subtopic labels, its
share of the page and a one-sentence description. *The data is not included in this repository.*

## Pipeline

| Step | Script | What it does | Uses API? |
|---|---|---|---|
| 1 | `step1_load.py` | Loads the database, parses each ID into issue date, page and object number, flags impossible dates | No |
| 2a | `step2a_group.py` | Groups objects by their existing Topic + Subtopic label; keeps the 717 groups seen in ≥50 issues (94.5% of objects); samples 24 example descriptions per group, spread across decades | No |
| 2b | `step2b_classify.py` | Sends each group's 24 examples to Claude, which gives it a clean standard series name and geographic level, and splits mixed groups into subtypes | Yes |
| 2c | `step2c_merge_names.py` | Merges names that describe the same series (1,614 → 1,309 canonical names) in one call | Yes |
| 2d | `step2d_assign.py` | Assigns every object a series: directly for coherent groups, by TF-IDF similarity to Claude's subtype examples for mixed groups | No |
| 3 | `step3_recurrence.py` | For each series: issues, first/last year, years active, coverage, issues per year → cadence, geography | No |
| 3b | `step3b_score.py` | Claude rates each frequent series (6 examples, spread across decades) as statistical / itemized / notice, its unit of observation and panel potential | Yes |
| 4 | `step4_validation_sample.py`, `step4b_accuracy.py` | Draws 5 random objects from each of the top 15 series for **hand-checking**, then computes accuracy | No |
| 5 | `step5_request_list.py` | For the series that were fully correct in the hand-check, picks one issue about every 5 years (1875–1945) and lists page numbers | No |
| 5b | `step5b_transit_request.py` | Request list for a bridges/subway topic: one issue per year 1878–1913 per theme (bridge traffic/tolls, rapid transit traffic, ward-level deaths and disease, vital statistics, building plans filed), matched by keywords | No |

API answers are cached in `data/`, so re-running a step does not call the API again.

## Results

- **1,290 recurring series** identified; 464 appear in ≥100 issues across ≥10 years.
- After scoring, **57 series have high panel potential** (98 classified as statistical data), e.g.
  Central Park weekly meteorological observations (1873–1937), Health Dept weekly vital statistics,
  Buildings Dept operations reports, Health Dept infectious disease reports (ward-week),
  bridge receipts and expenditures (bridge-month) and election returns by district.
- **Hand-check accuracy: 78.7%** (75 random objects from the top 15 series, judged strictly).
  Five series were fully correct: bridge receipts and expenditures, Buildings Dept operations reports,
  Health Dept infectious disease reports, Finance Dept and Sinking Fund statements, and Corporation
  Counsel opinions. Some labels are catch-alls that mix different tables (e.g. departmental budget
  appropriations, 40%) and need manual review.

## Data issues found

- 9,031 objects have impossible dates in their ID (e.g. `1908-02-31`, day `00`).
- 10,352 objects have a blank Type.
- Over half of the original Topic/Subtopic label groups mix different kinds of tables.
- Several series labelled "borough" start before the 1898 consolidation, so their geography likely
  changes over time (e.g. wards before 1898, boroughs after).

## Limitations

The pipeline works from the database's one-sentence descriptions, not the scans themselves, so it
produces a ranked map of where to look. Rare kinds of tables inside large label groups can be missed
by the 24-example sample (keyword searches, as in step 5b, help for specific topics). Cadence is
approximated by issues per active year, and the top candidates still need to be confirmed on the
actual pages.

## How to run

```bash
pip install -r requirements.txt
# put ANTHROPIC_API_KEY=... in a .env file (never commit it)
python step1_load.py
python step2a_group.py
python step2b_classify.py
python step2c_merge_names.py
python step2d_assign.py
python step3_recurrence.py
python step3b_score.py
python step4_validation_sample.py   # then fill in output/validation_sample.csv by hand (Y / N / U)
python step4b_accuracy.py
python step5_request_list.py
python step5b_transit_request.py
```

*Built with Claude as a coding assistant; pipeline design, validation and interpretation by Efekan Orsel.*
