import os
import re
import datetime as _dt
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


def _clean_text(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


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


def _rule_alert(conditions: str, allergies: str, medication: str, rules: list[dict]) -> tuple[int, str]:
    med_desc = _clean_text(medication)
    cond_text = _clean_text(conditions)
    all_text = _clean_text(allergies)

    # Allergy overlap (simple heuristic)
    for token in [t.strip() for t in re.split(r"[|,;]", all_text) if t.strip()]:
        if token and token in med_desc:
            return 1, f"Possible allergy match: allergy '{token}' overlaps medication"

    for rule in rules:
        if _contains_any(cond_text, rule["condition_keywords"]) and _contains_any(med_desc, rule["med_keywords"]):
            return 1, rule["reason"]

    return 0, ""


def _build_features(age, gender, conditions, allergies, medication) -> dict:
    return {
        "age": age,
        "gender": gender,
        "conditions": conditions,
        "allergies": allergies,
        "medication": medication,
        "text": f"{medication} || {conditions} || {allergies}",
    }


def _age_years(birthdate: str | None) -> int | None:
    if birthdate is None:
        return None
    s = str(birthdate).strip()
    if not s:
        return None
    try:
        bd = pd.to_datetime(s, errors="coerce")
        if pd.isna(bd):
            return None
        today = pd.Timestamp(_dt.date.today())
        delta = today.to_pydatetime() - bd.to_pydatetime()
        return round(max(0.0, delta.days / 365.25))
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_csv_bundle(base_dir: Path) -> dict[str, pd.DataFrame]:
    csv_dir = base_dir / "csv"
    patients = pd.read_csv(csv_dir / "patients.csv", dtype=str)
    conditions = pd.read_csv(csv_dir / "conditions.csv", dtype=str)
    allergies = pd.read_csv(csv_dir / "allergies.csv", dtype=str)
    meds = pd.read_csv(csv_dir / "medications.csv", dtype=str)

    for df in (patients, conditions, allergies, meds):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("")
    return {"patients": patients, "conditions": conditions, "allergies": allergies, "medications": meds}


def _patient_context(bundle: dict[str, pd.DataFrame], patient_id: str) -> tuple[list[str], list[str]]:
    cond_df = bundle["conditions"]
    all_df = bundle["allergies"]

    conds = (
        cond_df.loc[cond_df.get("PATIENT", "") == patient_id, "DESCRIPTION"]
        if "PATIENT" in cond_df.columns and "DESCRIPTION" in cond_df.columns
        else pd.Series([], dtype=str)
    )
    alls = (
        all_df.loc[all_df.get("PATIENT", "") == patient_id, "DESCRIPTION"]
        if "PATIENT" in all_df.columns and "DESCRIPTION" in all_df.columns
        else pd.Series([], dtype=str)
    )

    cond_list = sorted({_clean_text(x) for x in conds.astype(str).tolist() if _clean_text(x)})
    all_list = sorted({_clean_text(x) for x in alls.astype(str).tolist() if _clean_text(x)})
    return cond_list, all_list


def _recent_meds(bundle: dict[str, pd.DataFrame], patient_id: str, limit: int = 20) -> list[dict]:
    meds_df = bundle["medications"]
    if "PATIENT" not in meds_df.columns:
        return []

    rows = meds_df.loc[meds_df["PATIENT"] == patient_id].copy()
    if len(rows) == 0:
        return []

    if "START" in rows.columns:
        rows["_start"] = pd.to_datetime(rows["START"], errors="coerce")
        rows = rows.sort_values("_start", ascending=False)
    else:
        rows = rows.head(limit)

    out: list[dict] = []
    for _, r in rows.head(limit).iterrows():
        out.append(
            {
                "start": str(r.get("START", "")),
                "stop": str(r.get("STOP", "")),
                "description": str(r.get("DESCRIPTION", "")),
            }
        )
    return out


def create_app() -> Flask:
    app = Flask(__name__)

    base_dir = Path(__file__).resolve().parent
    env_model_path = os.environ.get("ALERT_MODEL_PATH", "").strip()
    if env_model_path:
        model_path = Path(env_model_path)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
    else:
        model_path = base_dir / "models" / "alert_model.joblib"
    threshold = float(os.environ.get("ALERT_THRESHOLD", "0.5"))

    model = None
    if model_path.exists():
        model = joblib.load(model_path)

    rules = _default_contraindication_rules()

    @app.get("/")
    def index():
        return render_template("index.html", model_loaded=bool(model), threshold=threshold)

    @app.get("/api/patients/search")
    def api_patients_search():
        q = str(request.args.get("q", "")).strip().lower()
        limit = 20
        if not q:
            return jsonify({"results": []})

        bundle = _load_csv_bundle(base_dir)
        p = bundle["patients"].copy()

        if "FIRST" in p.columns and "LAST" in p.columns:
            p["full_name"] = (p["FIRST"].astype(str).str.strip() + " " + p["LAST"].astype(str).str.strip()).str.strip()
        else:
            p["full_name"] = ""

        mask = p.get("Id", "").astype(str).str.lower().str.contains(q, na=False)
        if "full_name" in p.columns:
            mask = mask | p["full_name"].astype(str).str.lower().str.contains(q, na=False)
        p = p.loc[mask].head(limit)

        out: list[dict] = []
        for _, r in p.iterrows():
            pid = str(r.get("Id", ""))
            name = str(r.get("full_name", "")).strip()
            gender = str(r.get("GENDER", "")).strip()
            birthdate = str(r.get("BIRTHDATE", "")).strip()
            age = _age_years(birthdate)
            out.append(
                {
                    "id": pid,
                    "name": name,
                    "gender": gender,
                    "age": age,
                    "birthdate": birthdate,
                }
            )

        return jsonify({"results": out})

    @app.get("/api/patients/<patient_id>")
    def api_patient(patient_id: str):
        pid = str(patient_id).strip()
        bundle = _load_csv_bundle(base_dir)
        patients_df = bundle["patients"]
        match = patients_df.loc[patients_df.get("Id", "") == pid]
        if len(match) == 0:
            return jsonify({"error": "patient not found"}), 404

        r = match.iloc[0]
        full_name = (str(r.get("FIRST", "")).strip() + " " + str(r.get("LAST", "")).strip()).strip()
        birthdate = str(r.get("BIRTHDATE", "")).strip()
        gender = str(r.get("GENDER", "")).strip()
        age = _age_years(birthdate)

        conditions, allergies = _patient_context(bundle, pid)

        gender_prefill = gender[:1].upper() if gender else ""
        if gender_prefill not in {"M", "F"}:
            gender_prefill = ""

        return jsonify(
            {
                "id": pid,
                "name": full_name,
                "birthdate": birthdate,
                "age": age,
                "gender": gender_prefill,
                "conditions": "|".join(conditions),
                "allergies": "|".join(allergies),
            }
        )

    @app.get("/patients")
    def patients():
        q = str(request.args.get("q", "")).strip().lower()
        bundle = _load_csv_bundle(base_dir)
        p = bundle["patients"].copy()

        # Build display fields
        if "FIRST" in p.columns and "LAST" in p.columns:
            p["full_name"] = (p["FIRST"].astype(str).str.strip() + " " + p["LAST"].astype(str).str.strip()).str.strip()
        else:
            p["full_name"] = ""

        if q:
            mask = p.get("Id", "").astype(str).str.lower().str.contains(q, na=False)
            if "full_name" in p.columns:
                mask = mask | p["full_name"].astype(str).str.lower().str.contains(q, na=False)
            p = p.loc[mask]

        # Pagination
        per_page = 10
        total = len(p)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(int(request.args.get("page", 1)), total_pages))
        p = p.iloc[(page - 1) * per_page : page * per_page]

        patients_out: list[dict] = []
        for _, r in p.iterrows():
            pid = str(r.get("Id", ""))
            bd = str(r.get("BIRTHDATE", ""))
            age = _age_years(bd)
            patients_out.append(
                {
                    "id": pid,
                    "name": str(r.get("full_name", "")).strip(),
                    "gender": str(r.get("GENDER", "")).strip(),
                    "birthdate": bd,
                    "age": age,
                    "city": str(r.get("CITY", "")).strip(),
                    "state": str(r.get("STATE", "")).strip(),
                }
            )

        return render_template(
            "patients.html",
            patients=patients_out,
            q=q,
            page=page,
            total_pages=total_pages,
            total=total,
        )

    @app.get("/patients/<patient_id>")
    def patient_profile(patient_id: str):
        pid = str(patient_id).strip()
        bundle = _load_csv_bundle(base_dir)
        patients_df = bundle["patients"]

        match = patients_df.loc[patients_df.get("Id", "") == pid]
        if len(match) == 0:
            return render_template("patient_profile.html", not_found=True, patient_id=pid)

        r = match.iloc[0]
        full_name = (str(r.get("FIRST", "")).strip() + " " + str(r.get("LAST", "")).strip()).strip()
        birthdate = str(r.get("BIRTHDATE", "")).strip()
        gender = str(r.get("GENDER", "")).strip()
        age = _age_years(birthdate)

        conditions, allergies = _patient_context(bundle, pid)
        recent_meds = _recent_meds(bundle, pid, limit=20)

        patient = {
            "id": pid,
            "name": full_name,
            "gender": gender,
            "birthdate": birthdate,
            "age": age,
            "address": str(r.get("ADDRESS", "")).strip(),
            "city": str(r.get("CITY", "")).strip(),
            "state": str(r.get("STATE", "")).strip(),
            "zip": str(r.get("ZIP", "")).strip(),
        }

        # Prefill checker expects M/F; anything else becomes unknown in backend
        gender_prefill = gender[:1].upper() if gender else ""
        if gender_prefill not in {"M", "F"}:
            gender_prefill = ""

        return render_template(
            "patient_profile.html",
            not_found=False,
            patient=patient,
            conditions=conditions,
            allergies=allergies,
            recent_meds=recent_meds,
            model_loaded=bool(model),
            threshold=threshold,
            prefill={
                "age": ("" if age is None else str(int(age))),
                "gender": gender_prefill,
                "conditions": "|".join(conditions),
                "allergies": "|".join(allergies),
            },
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True) or {}

        age = payload.get("age", None)
        gender = payload.get("gender", "")
        conditions = payload.get("conditions", "")
        allergies = payload.get("allergies", "")
        medication = payload.get("medication", "")

        try:
            age_val = float(age) if age not in (None, "") else None
        except Exception:
            age_val = None

        gender_clean = str(gender).strip()[:1].upper()
        if gender_clean not in {"M", "F"}:
            gender_clean = "U"

        med_clean = _clean_text(medication)
        cond_clean = _clean_text(conditions)
        all_clean = _clean_text(allergies)

        if not med_clean:
            return jsonify({"error": "medication is required"}), 400

        rule_label, rule_reason = _rule_alert(cond_clean, all_clean, med_clean, rules)

        proba = None
        ml_label = None
        if model is not None:
            X = _build_features(age_val, gender_clean, cond_clean, all_clean, med_clean)
            X_df = pd.DataFrame([X])
            if X_df.loc[0, "age"] is None:
                X_df.loc[0, "age"] = float("nan")
            p = float(model.predict_proba(X_df)[0][1])
            proba = p
            ml_label = int(p >= threshold)

        final_label = int(rule_label == 1 or (ml_label == 1 if ml_label is not None else False))
        reason = rule_reason
        if final_label == 1 and not reason and ml_label == 1:
            reason = "ML risk score exceeded threshold"

        return jsonify(
            {
                "alert": final_label,
                "rule_alert": int(rule_label),
                "ml_alert": ml_label,
                "risk_score": proba,
                "threshold": threshold,
                "reason": reason,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
