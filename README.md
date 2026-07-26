# Medication Alert System (Flask + Rule-Based + ML)

A lightweight clinical-style dashboard that helps a prescriber check whether a proposed medication may be unsafe for a selected patient, using:

- Rule-based contraindication/allergy checks
- A trained ML model that returns a probability-like risk score and an ML alert

The UI supports:

- Patient directory and profiles (from Synthea CSVs)
- Patient search + auto-loading of patient context
- Medication checking workflow

## Features

- Medication checking page (`/`)
  - Search/select a patient (name or ID)
  - Enter proposed medication
  - View rule-based alert + ML risk score
- Patient directory (`/patients`)
  - Search and open a patient profile
- Patient profile (`/patients/<id>`)
  - Demographics, conditions, allergies, recent medications (from CSVs)

## Tech Stack

- Backend: Python + Flask
- ML: scikit-learn (logistic regression pipeline) + joblib
- Data: Synthea synthetic EHR exports (CSV)
- Frontend: Jinja templates + vanilla JavaScript + CSS

## Project Structure

- `app.py` Flask app (routes + prediction logic)
- `templates/` UI pages
- `static/styles.css` UI styling
- `static/app.js` UI behavior (patient search, predict, fill example, clear)
- `csv/` Synthea CSVs (patients/conditions/allergies/medications)
- `models/alert_model.joblib` trained ML pipeline
- `train_model.py` trains and saves the ML model

## Setup

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Then open:

- `http://127.0.0.1:5000/`

## API Endpoints

- `POST /predict`
  - Content-Type: `application/json`
  - Body:

```json
{
  "age": 32,
  "gender": "F",
  "conditions": "pregnancy",
  "allergies": "latex allergy",
  "medication": "warfarin"
}
```

- `GET /api/patients/search?q=<query>`
- `GET /api/patients/<id>`

## Demo Test Cases

Use these as quick sanity checks:

1) Pregnancy + warfarin
- **Input**: `conditions = pregnancy`, `medication = warfarin`
- **Expected**: Rule alert should trigger (contraindication), alert = 1

2) Hypertension + pseudoephedrine
- **Input**: `conditions = hypertension`, `medication = pseudoephedrine`
- **Expected**: Rule alert should trigger (contraindication), alert = 1

3) No known risk combination
- **Input**: `conditions = seasonal allergies`, `medication = loratadine`
- **Expected**: Likely no rule alert, alert may be 0 (depends on ML)

## Notes

- The dataset is synthetic (Synthea) and is used for academic/demo purposes.
- The ML model file is expected at `models/alert_model.joblib`.

## License

Add your preferred license here (or remove this section).
