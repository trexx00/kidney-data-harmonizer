"""Profile the sample files in data/samples/.

Prints quick facts about each source so you can write docs/DATA_PROFILE.md.
Nothing here cleans or changes the data. It only looks.

Run from the repo root with the virtual environment active:
    python scripts/profile_samples.py
"""

import json
from pathlib import Path

import pandas as pd

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
CTGOV_FILE = SAMPLES / "ctgov_sample.json"
OPENFDA_FILE = SAMPLES / "openfda_sample.json"


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# ClinicalTrials.gov
# ---------------------------------------------------------------------------

def profile_ctgov(path):
    header(f"ClinicalTrials.gov: {path.name}")
    studies = load_json(path)["studies"]

    # One row per trial. Fields live under protocolSection.<module>.<field>.
    trials = pd.json_normalize(studies)
    print(f"Trials: {len(trials)}")

    p = "protocolSection."
    ids = trials[p + "identificationModule.nctId"]
    print(f"Distinct NCT IDs: {ids.nunique()}")

    # Status values
    print("\nOverall status counts:")
    print(trials[p + "statusModule.overallStatus"].value_counts().to_string())

    # Problem: dates can be full (2022-06-01) or month-only (2022-06)
    start = trials[p + "statusModule.startDateStruct.date"].fillna("")
    print("\nStart date formats (by string length; 10 = YYYY-MM-DD, 7 = YYYY-MM):")
    print(start.str.len().value_counts().to_string())
    print("Examples:", start.head(5).tolist())

    # Problem: multiple conditions per trial
    n_conditions = trials[p + "conditionsModule.conditions"].apply(
        lambda c: len(c) if isinstance(c, list) else 0
    )
    print("\nConditions per trial:")
    print(n_conditions.value_counts().sort_index().to_string())

    # One row per intervention. Some trials have no interventions list.
    interventions = pd.json_normalize(
        studies,
        record_path=["protocolSection", "armsInterventionsModule", "interventions"],
        meta=[["protocolSection", "identificationModule", "nctId"]],
        errors="ignore",
    )
    missing = sum(
        1 for s in studies
        if not s["protocolSection"].get("armsInterventionsModule", {}).get("interventions")
    )
    print(f"\nInterventions: {len(interventions)} (trials with none listed: {missing})")

    # Problem: many interventions are not drugs
    print("\nIntervention types:")
    print(interventions["type"].value_counts().to_string())

    # Problem: placebo arms listed as interventions
    placebo = interventions["name"].str.contains("placebo", case=False, na=False)
    print(f"\nInterventions with 'placebo' in the name: {placebo.sum()}")

    # Raw drug names, to spot case, dose text and combinations
    drugs = interventions.loc[interventions["type"] == "DRUG", "name"]
    print("\nSample raw DRUG intervention names:")
    for name in drugs.drop_duplicates().head(15):
        print("  ", repr(name))

    return trials, interventions


# ---------------------------------------------------------------------------
# openFDA adverse events (FAERS)
# ---------------------------------------------------------------------------

def profile_openfda(path):
    header(f"openFDA adverse events: {path.name}")
    data = load_json(path)
    results = data["results"]
    total = data["meta"]["results"]["total"]
    print(f"Reports in sample: {len(results)} (total matching the query: {total})")

    reports = pd.json_normalize(results)

    # Problem: duplicates / multiple versions of a report
    ids = reports["safetyreportid"]
    print(f"Distinct safetyreportid: {ids.nunique()}")
    print("safetyreportversion counts:")
    print(reports["safetyreportversion"].value_counts().to_string())
    if "duplicate" in reports:
        print(f"Reports flagged duplicate=1: {(reports['duplicate'] == '1').sum()}")

    # Problem: dates stored as text (YYYYMMDD)
    print("\nreceivedate examples (text, not dates):", reports["receivedate"].head(5).tolist())

    # Problem: age has a separate unit code (801=year, 802=month, 804=day ...)
    print("\nAge unit codes (patient.patientonsetageunit):")
    print(reports.get("patient.patientonsetageunit", pd.Series(dtype=str))
          .value_counts(dropna=False).to_string())
    age_missing = reports.get("patient.patientonsetage", pd.Series(dtype=str)).isna().sum()
    print(f"Reports with no age: {age_missing}")

    # One row per reported drug
    drugs = pd.json_normalize(
        results, record_path=["patient", "drug"], meta=["safetyreportid"]
    )
    print(f"\nDrug entries: {len(drugs)} "
          f"(avg {len(drugs) / len(results):.1f} per report)")

    # drugcharacterization: 1 = suspect, 2 = concomitant, 3 = interacting
    print("Drug role (drugcharacterization) counts:")
    print(drugs["drugcharacterization"].value_counts().to_string())

    # Problem: missing openfda.generic_name
    has_generic = drugs.get("openfda.generic_name", pd.Series(index=drugs.index)).notna()
    print(f"\nDrug entries missing openfda.generic_name: "
          f"{(~has_generic).sum()} of {len(drugs)} ({(~has_generic).mean():.0%})")

    # Problem: brand vs. generic names. Compare what the reporter typed with
    # openFDA's own brand_name / generic_name lists for that entry. A name in
    # both lists (e.g. generics sold under their generic name) counts as generic.
    def list_col(col):
        s = drugs.get(col, pd.Series(index=drugs.index, dtype=object))
        return s.apply(lambda v: {x.upper().strip() for x in v} if isinstance(v, list) else set())

    typed = drugs["medicinalproduct"].fillna("").str.upper().str.strip().str.rstrip(".")
    brands, generics = list_col("openfda.brand_name"), list_col("openfda.generic_name")
    name_kind = pd.Series("typed name not in openFDA lists", index=drugs.index)
    name_kind[[t in b for t, b in zip(typed, brands)]] = "brand name"
    name_kind[[t in g for t, g in zip(typed, generics)]] = "generic name"
    name_kind[~has_generic & (brands.str.len() == 0)] = "no openFDA match (can't tell)"
    print("\nWhat the reporter typed, checked against openFDA brand/generic lists:")
    print(name_kind.value_counts().to_string())
    print("Examples typed as a brand name:",
          sorted(set(typed[name_kind == "brand name"]))[:10])

    print("\nSample raw medicinalproduct names (as typed by the reporter):")
    for name in drugs["medicinalproduct"].drop_duplicates().head(15):
        print("  ", repr(name))

    print("\nSample entries WITHOUT a generic name:")
    for name in drugs.loc[~has_generic, "medicinalproduct"].drop_duplicates().head(10):
        print("  ", repr(name))

    # Reactions: one row per reaction term
    reactions = pd.json_normalize(
        results, record_path=["patient", "reaction"], meta=["safetyreportid"]
    )
    print(f"\nReaction entries: {len(reactions)}")
    print("Top reaction terms:")
    print(reactions["reactionmeddrapt"].value_counts().head(10).to_string())

    return reports, drugs, reactions


# ---------------------------------------------------------------------------
# Cross-source look: do the names line up at all?
# ---------------------------------------------------------------------------

def compare_names(interventions, drugs):
    header("Cross-source: naive name overlap (lowercase, trimmed)")
    trial_names = set(
        interventions.loc[interventions["type"] == "DRUG", "name"]
        .str.lower().str.strip()
    )
    fda_names = set(drugs["medicinalproduct"].dropna().str.lower().str.strip())
    overlap = trial_names & fda_names
    print(f"Distinct trial drug names: {len(trial_names)}")
    print(f"Distinct FAERS drug names: {len(fda_names)}")
    print(f"Exact overlap: {len(overlap)} -> {sorted(overlap)[:10]}")
    print("(Low overlap is expected; it's why Session 3 uses RxNorm.)")


def main():
    _, interventions = profile_ctgov(CTGOV_FILE)
    _, drugs, _ = profile_openfda(OPENFDA_FILE)
    compare_names(interventions, drugs)


if __name__ == "__main__":
    main()
