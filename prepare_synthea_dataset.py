import argparse
import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _clean_text(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_date(s: str):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return pd.NaT
    s = str(s).strip()
    if not s:
        return pd.NaT
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(ts):
            return pd.NaT
        return ts.tz_convert(None)
    except Exception:
        return pd.NaT


def _age_at(birthdate, when) -> float:
    if pd.isna(birthdate) or pd.isna(when):
        return float("nan")
    if isinstance(birthdate, str):
        birthdate = _parse_date(birthdate)
    if isinstance(when, str):
        when = _parse_date(when)
    if pd.isna(birthdate) or pd.isna(when):
        return float("nan")
    delta = when.to_pydatetime() - birthdate.to_pydatetime()
    return max(0.0, delta.days / 365.25)


def _contains_any(haystack: str, needles: list[str]) -> bool:
    h = _clean_text(haystack)
    return any(n in h for n in needles)


def _default_contraindication_rules() -> list[dict]:
    return [
        {
            "condition_keywords": ["ulcer", "peptic"],
            "med_keywords": ["ibuprofen", "naproxen", "diclofenac", "aspirin", "indomethacin", "ketorolac"],
            "reason": "Possible contraindication: ulcer/peptic disease + NSAID risk",
        },
        {
            "condition_keywords": ["asthma"],
            "med_keywords": ["propranolol", "nadolol", "timolol"],
            "reason": "Possible contraindication: asthma + non-selective beta blocker risk",
        },
        {
            "condition_keywords": ["pregnan", "pregnancy"],
            "med_keywords": ["warfarin", "isotretinoin", "lisinopril", "enalapril", "valpro"],
            "reason": "Possible contraindication: pregnancy + teratogenic/unsafe medication risk",
        },
        {
            "condition_keywords": ["kidney", "renal"],
            "med_keywords": ["ibuprofen", "naproxen", "diclofenac", "ketorolac"],
            "reason": "Possible contraindication: kidney/renal disease + NSAID risk",
        },
        {
            "condition_keywords": ["hypertension"],
            "med_keywords": ["pseudoephedrine"],
            "reason": "Possible contraindication: hypertension + decongestant risk",
        },
        {
            "condition_keywords": ["diabetes"],
            "med_keywords": ["hydrochlorothiazide", "furosemide"],
            "reason": "Possible contraindication: diabetes + diuretic risk",
        },
    ]


def _load_rules_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    required = {"condition_keywords", "med_keywords", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rules CSV missing columns: {sorted(missing)}")

    rules: list[dict] = []
    for _, row in df.iterrows():
        cond = [x.strip().lower() for x in str(row["condition_keywords"]).split("|") if str(x).strip()]
        meds = [x.strip().lower() for x in str(row["med_keywords"]).split("|") if str(x).strip()]
        reason = str(row["reason"]).strip()
        if cond and meds and reason:
            rules.append({"condition_keywords": cond, "med_keywords": meds, "reason": reason})
    return rules


def _make_patient_context(conditions: pd.DataFrame, allergies: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    cond_map: dict[str, set[str]] = {}
    all_map: dict[str, set[str]] = {}

    if len(conditions) > 0:
        for pid, g in conditions.groupby("PATIENT"):
            conds = set(_clean_text(x) for x in g["DESCRIPTION"].astype(str).tolist() if str(x).strip())
            cond_map[str(pid)] = {c for c in conds if c}

    if len(allergies) > 0:
        for pid, g in allergies.groupby("PATIENT"):
            alls = set(_clean_text(x) for x in g["DESCRIPTION"].astype(str).tolist() if str(x).strip())
            all_map[str(pid)] = {a for a in alls if a}

    return cond_map, all_map


def _apply_rules(med_desc: str, patient_conditions: set[str], patient_allergies: set[str], rules: list[dict]) -> tuple[int, str]:
    med_desc_clean = _clean_text(med_desc)

    allergy_label = 0
    allergy_reason = ""
    for a in patient_allergies:
        a_clean = _clean_text(a)
        if not a_clean:
            continue
        if a_clean in med_desc_clean:
            allergy_label = 1
            allergy_reason = f"Possible allergy match: allergy '{a_clean}' overlaps medication"
            break

    contra_label = 0
    contra_reason = ""
    patient_cond_text = " ".join(sorted(patient_conditions))
    for rule in rules:
        if _contains_any(patient_cond_text, rule["condition_keywords"]) and _contains_any(med_desc_clean, rule["med_keywords"]):
            contra_label = 1
            contra_reason = rule["reason"]
            break

    if allergy_label or contra_label:
        reason = allergy_reason if allergy_label else contra_reason
        if allergy_label and contra_label:
            reason = allergy_reason + "; " + contra_reason
        return 1, reason

    return 0, ""


def build_dataset(input_dir: Path, rules_csv: Path | None) -> pd.DataFrame:
    patients = pd.read_csv(input_dir / "patients.csv")
    conditions = pd.read_csv(input_dir / "conditions.csv")
    medications = pd.read_csv(input_dir / "medications.csv")
    allergies = pd.read_csv(input_dir / "allergies.csv")

    patients_required = {"Id", "BIRTHDATE", "GENDER"}
    missing = patients_required - set(patients.columns)
    if missing:
        raise ValueError(f"patients.csv missing columns: {sorted(missing)}")

    for df, name in [(conditions, "conditions.csv"), (medications, "medications.csv"), (allergies, "allergies.csv")]:
        if "PATIENT" not in df.columns:
            raise ValueError(f"{name} missing column: PATIENT")
        if "DESCRIPTION" not in df.columns:
            raise ValueError(f"{name} missing column: DESCRIPTION")

    if "START" not in medications.columns:
        raise ValueError("medications.csv missing column: START")

    patients = patients.copy()
    patients["Id"] = patients["Id"].astype(str)
    patients["BIRTHDATE"] = patients["BIRTHDATE"].map(_parse_date)

    medications = medications.copy()
    medications["PATIENT"] = medications["PATIENT"].astype(str)
    medications["START"] = medications["START"].map(_parse_date)
    medications = medications.dropna(subset=["PATIENT", "START", "DESCRIPTION"])

    cond_map, all_map = _make_patient_context(conditions, allergies)

    rules = _load_rules_csv(rules_csv) if rules_csv else _default_contraindication_rules()

    rows: list[dict] = []
    for _, med in medications.iterrows():
        pid = str(med["PATIENT"])
        med_desc = str(med["DESCRIPTION"])
        start = med["START"]

        p = patients.loc[patients["Id"] == pid]
        if len(p) == 0:
            continue
        birthdate = p.iloc[0]["BIRTHDATE"]
        gender = str(p.iloc[0]["GENDER"])
        age = _age_at(birthdate, start)

        p_conditions = cond_map.get(pid, set())
        p_allergies = all_map.get(pid, set())

        label, reason = _apply_rules(med_desc, p_conditions, p_allergies, rules)

        rows.append(
            {
                "patient_id": pid,
                "age": age,
                "gender": gender,
                "medication": _clean_text(med_desc),
                "conditions": "|".join(sorted(p_conditions)),
                "allergies": "|".join(sorted(p_allergies)),
                "label": label,
                "reason": reason,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["age", "gender", "medication"])
    df = df[df["medication"].astype(str).str.len() > 0]
    df = df[df["age"].apply(lambda x: pd.notna(x) and x >= 0)]
    df = df.drop_duplicates(subset=["patient_id", "medication", "conditions", "allergies"])

    return df


def build_simulated_dataset(
    input_dir: Path,
    rules_csv: Path | None,
    samples_per_patient: int,
    seed: int,
) -> pd.DataFrame:
    patients = pd.read_csv(input_dir / "patients.csv")
    conditions = pd.read_csv(input_dir / "conditions.csv")
    allergies = pd.read_csv(input_dir / "allergies.csv")

    patients_required = {"Id", "BIRTHDATE", "GENDER"}
    missing = patients_required - set(patients.columns)
    if missing:
        raise ValueError(f"patients.csv missing columns: {sorted(missing)}")

    for df, name in [(conditions, "conditions.csv"), (allergies, "allergies.csv")]:
        if "PATIENT" not in df.columns:
            raise ValueError(f"{name} missing column: PATIENT")
        if "DESCRIPTION" not in df.columns:
            raise ValueError(f"{name} missing column: DESCRIPTION")

    patients = patients.copy()
    patients["Id"] = patients["Id"].astype(str)
    patients["BIRTHDATE"] = patients["BIRTHDATE"].map(_parse_date)

    cond_map, all_map = _make_patient_context(conditions, allergies)
    rules = _load_rules_csv(rules_csv) if rules_csv else _default_contraindication_rules()

    risky_meds: list[str] = sorted({m for r in rules for m in r["med_keywords"]})
    safe_meds: list[str] = ["acetaminophen", "amoxicillin", "saline nasal spray", "vitamin c"]
    if not risky_meds:
        risky_meds = ["ibuprofen"]

    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(_dt.datetime(2020, 1, 1))

    rows: list[dict] = []
    for pid in patients["Id"].astype(str).tolist():
        p = patients.loc[patients["Id"] == pid].iloc[0]
        age = _age_at(p["BIRTHDATE"], as_of)
        gender = str(p["GENDER"])

        p_conditions = cond_map.get(pid, set())
        p_allergies = all_map.get(pid, set())

        for med in safe_meds:
            label, reason = _apply_rules(med, p_conditions, p_allergies, rules)
            rows.append(
                {
                    "patient_id": pid,
                    "age": age,
                    "gender": gender,
                    "medication": _clean_text(med),
                    "conditions": "|".join(sorted(p_conditions)),
                    "allergies": "|".join(sorted(p_allergies)),
                    "label": int(label),
                    "reason": reason,
                }
            )

        for _ in range(max(1, samples_per_patient)):
            med = str(rng.choice(risky_meds))
            label, reason = _apply_rules(med, p_conditions, p_allergies, rules)
            rows.append(
                {
                    "patient_id": pid,
                    "age": age,
                    "gender": gender,
                    "medication": _clean_text(med),
                    "conditions": "|".join(sorted(p_conditions)),
                    "allergies": "|".join(sorted(p_allergies)),
                    "label": int(label),
                    "reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["age", "gender", "medication"])
    df = df[df["medication"].astype(str).str.len() > 0]
    df = df[df["age"].apply(lambda x: pd.notna(x) and x >= 0)]
    df = df.drop_duplicates(subset=["patient_id", "medication", "conditions", "allergies"])

    return df


def split_and_save(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if df["label"].nunique() < 2:
        df.to_csv(out_dir / "dataset.csv", index=False)
        summary = pd.DataFrame(
            {
                "split": ["all"],
                "rows": [len(df)],
                "alert_rate(label=1)": [float(df["label"].mean()) if len(df) else 0.0],
            }
        )
        summary.to_csv(out_dir / "summary.csv", index=False)
        return

    rng = np.random.default_rng(seed)

    def _stratified_split(frame: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        parts_a: list[pd.DataFrame] = []
        parts_b: list[pd.DataFrame] = []
        for _, g in frame.groupby("label"):
            g = g.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            n_b = max(1, int(round(len(g) * test_size)))
            parts_b.append(g.iloc[:n_b])
            parts_a.append(g.iloc[n_b:])
        a = pd.concat(parts_a, ignore_index=True)
        b = pd.concat(parts_b, ignore_index=True)
        a = a.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        b = b.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        return a, b

    train_df, temp_df = _stratified_split(df, test_size=0.2)
    val_df, test_df = _stratified_split(temp_df, test_size=0.5)

    df.to_csv(out_dir / "dataset.csv", index=False)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    summary = pd.DataFrame(
        {
            "split": ["all", "train", "val", "test"],
            "rows": [len(df), len(train_df), len(val_df), len(test_df)],
            "alert_rate(label=1)": [
                float(df["label"].mean()),
                float(train_df["label"].mean()),
                float(val_df["label"].mean()),
                float(test_df["label"].mean()),
            ],
        }
    )
    summary.to_csv(out_dir / "summary.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Folder containing Synthea CSV exports")
    p.add_argument("--out-dir", default="data/synthea_processed", help="Output folder")
    p.add_argument("--rules-csv", default="", help="Optional contraindication rules CSV")
    p.add_argument(
        "--mode",
        default="simulate",
        choices=["simulate", "events"],
        help="simulate: create dataset for checking a proposed prescription; events: use raw medication events",
    )
    p.add_argument(
        "--samples-per-patient",
        type=int,
        default=3,
        help="Number of risky medication samples per patient (simulate mode)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    required_files = ["patients.csv", "conditions.csv", "medications.csv", "allergies.csv"]
    missing = [f for f in required_files if not (input_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Synthea files in input dir: {missing}")

    rules_csv = Path(args.rules_csv) if args.rules_csv.strip() else None
    if rules_csv is not None and not rules_csv.exists():
        raise FileNotFoundError("rules-csv path does not exist")

    if args.mode == "events":
        df = build_dataset(input_dir=input_dir, rules_csv=rules_csv)
    else:
        df = build_simulated_dataset(
            input_dir=input_dir,
            rules_csv=rules_csv,
            samples_per_patient=args.samples_per_patient,
            seed=args.seed,
        )
    split_and_save(df, Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
