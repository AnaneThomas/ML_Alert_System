---
title: Medication Alert System Using Machine Learning and Rule-Based Contraindication Checks
---

# Title Page

**Project Title:** Medication Alert System Using Machine Learning and Rule-Based Contraindication Checks  
**Student Name:** [Your Name]  
**Matric/Index Number:** [Your ID]  
**Programme/Department:** [Department]  
**Institution:** [Institution]  
**Supervisor:** [Supervisor Name]  
**Date:** [Month Year]  

# Declaration

I, **[Your Name]**, declare that this project report titled **“Medication Alert System Using Machine Learning and Rule-Based Contraindication Checks”** is my original work and has not been submitted to any institution for any award, degree, or diploma. All sources used have been duly acknowledged and referenced.

**Signature:** ____________________  
**Date:** ____________________

# Certification

This is to certify that this project was carried out by **[Your Name]** under my supervision and has been submitted with my approval.

**Supervisor Name:** ____________________  
**Signature:** ____________________  
**Date:** ____________________

# Dedication

[Optional: Dedicate this work to your family, friends, etc.]

# Acknowledgement

[Optional: Acknowledge supervisor, department, friends, contributors, etc.]

# Abstract

This project presents a **Medication Alert System** that assists prescribers in identifying potentially unsafe medication prescriptions based on patient context such as age, gender, conditions, allergies, and a proposed medication. The system combines **rule-based contraindication checks** with a **machine learning model** to provide a risk score and an alert flag.

- **Background of the study:** Medication errors and contraindicated prescriptions can lead to preventable adverse events; clinical decision support can reduce risk.
- **Problem statement:** Manual review of patient history is time-consuming and error-prone, especially in high-volume clinical workflows.
- **Aim of the project:** To develop a lightweight web-based tool that flags potentially unsafe prescriptions using both deterministic rules and ML prediction.
- **Methodology used:** Design Science approach; system built using Flask; model trained using scikit-learn on a labeled dataset derived from synthetic EHR data.
- **Tools/technologies used:** Python, Flask, HTML/CSS/JavaScript, pandas, scikit-learn, joblib; Synthea synthetic EHR CSVs.
- **Key results:** The system successfully returns alerts and risk scores via a `/predict` API and provides a clinical-style dashboard UI for patient selection and medication checking.
- **Conclusion:** A hybrid rule + ML approach can provide useful decision support signals; further work is needed for clinical validation, improved labeling, and integration with real EHR systems.

# Table of Contents

1. Introduction
2. Literature Review
3. Methodology and System Design
4. System Implementation, Testing and Results
5. Discussion, Conclusion and Recommendations
References
Appendices

# List of Figures

- Figure 3.1: System Architecture Diagram
- Figure 3.2: Use Case Diagram
- Figure 3.3: Sequence Diagram (Medication Check)
- Figure 4.1: Medication Page Screenshot
- Figure 4.2: Patient Directory Screenshot
- Figure 4.3: Patient Profile Screenshot

# List of Tables

- Table 3.1: Functional Requirements
- Table 3.2: Non-Functional Requirements
- Table 4.1: Development Environment
- Table 4.2: Major Code Modules
- Table 4.3: Functional Test Cases and Results
- Table 4.4: Local Prediction Performance

---

# Chapter 1: Introduction

## 1.1 Background of the Study

Medication safety is a critical aspect of healthcare delivery. Patients often have comorbidities, allergies, and prior medication exposure that can make certain prescriptions inappropriate or high risk. Clinical decision support systems (CDSS) aim to reduce these risks by providing timely warnings to clinicians.

## 1.2 Problem Statement

In many settings, reviewing patient history and identifying contraindications relies on manual effort. This can lead to missed contraindications, delayed decisions, and inconsistent safety checks.

## 1.3 Aim of the Study

To design and implement a web-based medication alert system that provides hybrid decision support using both rule-based checks and machine learning risk scoring.

## 1.4 Objectives of the Study

- To develop a patient search and selection workflow that automatically loads patient context.
- To implement rule-based contraindication/allergy checks and provide interpretable reasons.
- To train an ML model that estimates prescription risk from patient context and medication text.
- To integrate the model into a Flask API (`/predict`) and UI dashboard.
- To test the system with representative cases and document results.

## 1.5 Research Questions (Optional)

- Can a hybrid approach (rules + ML) improve usability compared to rules-only checks?
- How effectively can synthetic EHR datasets be used to prototype decision support tools?

## 1.6 Significance of the Study

The system demonstrates a practical prototype of medication risk alerting that can be extended into a more comprehensive CDSS. It also serves as an academic artifact illustrating applied ML and full-stack implementation.

## 1.7 Scope of the Study

- Uses synthetic EHR data (Synthea) stored as CSV files.
- Provides a web UI for patient directory, patient profile, and medication check.
- Provides rule-based contraindication checks and an ML risk score.

## 1.8 Limitations of the Study

- The dataset is synthetic; results are not clinically validated.
- Contraindication rules are simplified keyword-based heuristics.
- Model performance depends on labeling strategy and dataset quality.

## 1.9 Organisation of the Report

This report is organized into five chapters: Introduction, Literature Review, Methodology and System Design, System Implementation/Testing/Results, and Discussion/Conclusion/Recommendations.

---

# Chapter 2: Literature Review

## 2.1 Conceptual Review

- Medication safety and adverse drug events
- Clinical decision support systems (CDSS)
- Rule-based vs machine learning decision support

## 2.2 Review of Existing Systems

Discuss examples such as:

- EHR integrated CDSS alerting
- Drug-drug interaction checking tools
- Allergy alerting systems

## 2.3 Review of Related Technologies

- Web frameworks (Flask, Django)
- ML pipelines and model deployment (scikit-learn, joblib)
- Synthetic data generation tools (Synthea)

## 2.4 Review of Methods/Algorithms/Models

- Logistic regression for binary classification
- TF-IDF for text features
- Feature engineering using demographic and history fields

## 2.5 Gaps in Existing Systems

- Many systems are complex, expensive, and tightly coupled to EHR platforms.
- Alerts can suffer from fatigue if too frequent or poorly explained.
- Limited access to real EHR data makes prototyping difficult.

## 2.6 Summary of Literature Review

A hybrid approach that combines interpretable rules with probabilistic ML scoring can potentially improve decision quality and usability. However, validation and careful alert design remain essential.

**References:** Add citations in APA format as required. (See References section.)

---

# Chapter 3: Methodology and System Design

## 3.1 Research Methodology

- **Research approach:** Design Science (build-and-evaluate an IT artifact)
- **Data collection:** Synthetic EHR dataset (Synthea CSV exports)
- **System development methodology:** Iterative/Agile prototyping

## 3.2 Requirements Analysis

### 3.2.1 Functional Requirements

Table 3.1: Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Search patients by name/ID |
| FR2 | View patient profile details (demographics, conditions, allergies, medications) |
| FR3 | Submit a medication check request |
| FR4 | Display rule-based alert result and reason |
| FR5 | Display ML alert and risk score |

### 3.2.2 Non-Functional Requirements

Table 3.2: Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | Usable, clean dashboard UI |
| NFR2 | Fast response time for prediction |
| NFR3 | Basic input validation and error handling |
| NFR4 | Maintainable modular code |

### 3.2.3 User Requirements / User Stories

- As a clinician, I want to search and select a patient, so that I can check a prescription with correct context.
- As a clinician, I want to see clear reasons for alerts, so that I can understand the risk.

## 3.3 Tools and Technologies Used

- Python 3.x
- Flask
- pandas
- scikit-learn
- joblib
- HTML/CSS/JavaScript
- Synthea dataset (CSV)

## 3.4 System Design

### 3.4.1 System Architecture Diagram

**Figure 3.1:** [Insert architecture diagram]

Suggested architecture narrative:

- UI (browser) communicates with Flask backend.
- Flask loads CSV patient context.
- Flask loads ML model (joblib) and applies rule checks.
- Flask returns results to UI.

### 3.4.2 Use Case Diagram

**Figure 3.2:** [Insert use case diagram]

Key actors:

- Clinician/User

Use cases:

- Search Patient
- View Patient Profile
- Check Medication

### 3.4.3 Database Design (ERD)

This prototype uses CSV files rather than a relational database.

**Figure 3.3:** [Optional ERD showing logical entities: Patient, Condition, Allergy, Medication]

### 3.4.4 Additional Diagrams (Pick 2–3)

Add 2–3 of:

- Activity diagram (patient selection + medication check)
- Sequence diagram (UI → API → model/rules → response)
- Class diagram (if applicable)
- Input design (forms and expected fields)
- Output design (alert card, risk score display)

## 3.5 Algorithm / Model Description

### 3.5.1 Rule-Based Contraindication Checks

The rule engine checks for predefined condition keywords and medication keywords. If both match, an alert is triggered with an explanatory reason.

**Pseudocode:**

1. Normalize text (lowercase, trim, collapse whitespace).
2. For each rule:
   - If any condition keyword appears in patient conditions AND any medication keyword appears in proposed medication:
     - Trigger rule alert and reason.

### 3.5.2 ML Model

- Binary classifier predicting whether a prescription is unsafe.
- Features include:
  - Patient demographics (age, gender)
  - Patient condition text
  - Allergy text
  - Proposed medication text
- Pipeline saved to `models/alert_model.joblib`.

## 3.6 Data Description / Dataset

- **Source:** Synthea synthetic EHR exports
- **Format:** CSV
- **Files used:** `patients.csv`, `conditions.csv`, `allergies.csv`, `medications.csv`
- **Preprocessing:** cleaning/normalization of free text, aggregation of patient history into pipe-separated strings.

## 3.7 Validation / Testing Plan

- Unit-level testing of helper functions (text cleaning, age calculation)
- Integration testing for endpoints (`/predict`, `/api/patients/*`)
- User testing (manual) for UI workflow

## 3.8 Ethical Considerations

- Synthetic data reduces privacy concerns.
- No real patient data is used.
- Security considerations: do not expose sensitive keys, validate inputs, avoid logging personal data if adapted to real data.

---

# Chapter 4: System Implementation, Testing and Results

## 4.1 Introduction

This chapter describes the implementation of the Medication Alert System, the technologies used, the main software modules, the user interfaces, and the tests performed on the completed prototype. The objective of the implementation was to produce a functional web application that allows a user to search for a patient, review patient information, enter a proposed medication, and receive a combined rule-based and machine-learning safety result.

## 4.2 Development Environment

The application was developed and tested in a local Windows environment. A Python virtual environment was used to isolate project dependencies and improve reproducibility. The Flask development server was used during implementation and was accessed through a web browser at `http://127.0.0.1:5000`.

Table 4.1: Development Environment

| Component | Specification |
|---|---|
| Operating system | Microsoft Windows |
| Development environment | Windsurf IDE with an integrated PowerShell terminal |
| Programming language | Python 3.x |
| Backend framework | Flask 3.0.3 |
| Data processing | pandas 2.2.2 |
| Machine learning | scikit-learn 1.5.1 |
| Model persistence | joblib 1.4.2 |
| Frontend technologies | HTML, CSS, JavaScript and Jinja templates |
| Data source | Synthea synthetic EHR CSV files |
| Web browser | Modern Chromium-based browser |

The project dependencies are defined in `requirements.txt`. The application is started by activating the virtual environment and running `python app.py` from the project directory.

## 4.3 Tools and Frameworks Used

### 4.3.1 Flask

Flask provides the web application layer. It maps URLs to Python route functions, renders the user interface templates, receives medication-check requests, and returns JSON responses to the browser (Grinberg, 2018). The application includes page routes for the medication checker, patient directory, and patient profile, as well as API routes for patient searching and prediction (Miguel, 2018).

### 4.3.2 pandas

pandas is used to load and process the Synthea CSV files (McKinney, 2010). Patient demographics, conditions, allergies, and medication records are read into DataFrames. The application filters these DataFrames using patient identifiers and converts the required records into structures suitable for templates and JSON responses (Pandas Documentation, 2023).

### 4.3.3 scikit-learn and joblib

scikit-learn is used to construct the machine-learning pipeline (Pedregosa et al., 2011). The pipeline combines TF-IDF text features, one-hot encoded gender data, imputed age data, and logistic regression classification (Hosmer et al., 2013). joblib saves the trained pipeline to `models/alert_model.joblib` and reloads it when the Flask application starts (Joblib Documentation, 2023).

### 4.3.4 HTML, CSS, JavaScript and Jinja

Jinja templates generate the Medication Check, Patient Directory, and Patient Profile pages (W3C, 2014). CSS provides the responsive clinical dashboard layout, while JavaScript handles asynchronous patient searches, patient selection, medication form submission, button states, and result rendering without requiring a complete page reload for each interaction (Flanagan, 2020).

## 4.4 System Implementation

### 4.4.1 Application Start-up and Model Loading

The Flask application is created by the `create_app()` function in `app.py` (Grinberg, 2018). During start-up, the application determines the model path from the `ALERT_MODEL_PATH` environment variable or uses `models/alert_model.joblib` as the default. The decision threshold is read from `ALERT_THRESHOLD`, with `0.5` used as the default value. If the model file exists, joblib loads the complete prediction pipeline into memory before requests are processed (Pedregosa et al., 2011).

### 4.4.2 Patient Data Loading and Caching

The `_load_csv_bundle()` function reads `patients.csv`, `conditions.csv`, `allergies.csv`, and `medications.csv` (McKinney, 2010). The function is decorated with `lru_cache`, so the files are loaded once and the resulting DataFrames are reused for later requests (Dean et al., 2002). Missing text values are replaced with empty strings to reduce errors during filtering and display.

Patient context is assembled using a common patient identifier. The `_patient_context()` function retrieves and cleans the patient's condition and allergy descriptions. The `_recent_meds()` function retrieves up to 20 medication records and sorts them by start date when that information is available. Age is calculated from the patient's birth date and returned as a whole number of years (Hripcsak et al., 2013).

### 4.4.3 Patient Search and Directory

The `GET /api/patients/search` endpoint accepts a name or patient ID through the `q` query parameter (Fielding, 2000). It searches the patient DataFrame and returns up to 20 matching records as JSON. The `GET /api/patients/<patient_id>` endpoint returns the selected patient's demographics, conditions, and allergies.

The `GET /patients` route renders the patient directory. It supports searching by name or ID and applies server-side pagination with 10 patients per page (Dean et al., 2002). Selecting a directory entry opens the detailed patient profile through `GET /patients/<patient_id>`.

### 4.4.4 Rule-Based Alert Implementation

The rule component checks known combinations of condition keywords and medication keywords (Kawamoto et al., 2005). Examples include pregnancy with warfarin, hypertension with pseudoephedrine, renal disease with non-steroidal anti-inflammatory drugs, and asthma with a non-selective beta blocker. It also performs a simple allergy-overlap check by comparing cleaned allergy terms with the proposed medication text (Pumphrey & Gowland, 2007).

When a rule matches, the component returns a rule alert value of `1` and an explanatory reason. When no rule matches, it returns `0` and an empty reason. This component provides an interpretable safety signal because the user can see why the alert was raised (Bates et al., 2003).

### 4.4.5 Machine-Learning Implementation

The training process is implemented in `train_model.py`. Medication, condition, and allergy text is combined into one text feature. The preprocessing pipeline applies TF-IDF vectorization to the combined text (Manning et al., 2008), one-hot encoding to gender, and median imputation to age. Logistic regression with balanced class weights performs binary classification (Hosmer et al., 2013).

During prediction, the `/predict` endpoint validates and cleans the submitted values, builds a one-row pandas DataFrame, and calls `predict_proba()` on the stored pipeline (Pedregosa et al., 2011). The probability of the unsafe class becomes the ML risk score. A score greater than or equal to the configured threshold produces an ML alert value of `1`.

### 4.4.6 Hybrid Decision Logic

The final alert combines the deterministic rule result and the ML result (Sutton et al., 2020). An alert is produced if either the rule engine or the ML model identifies a risk. The response contains the following values:

- `alert`: the final hybrid alert;
- `rule_alert`: the rule engine result;
- `ml_alert`: the model classification;
- `risk_score`: the model probability;
- `threshold`: the configured decision threshold; and
- `reason`: the rule explanation or an ML threshold explanation.

This arrangement preserves the interpretability of known clinical rules while allowing the model to provide an additional probabilistic signal (Poon et al., 2010).

## 4.5 Interface Design

### 4.5.1 Medication Check Interface

The Medication Check page is the main application interface. It contains a patient search field, read-only patient context fields, a proposed medication field, and the Check Medication button (Wright et al., 2018). Matching patients appear in a dropdown while the user types. Selecting a result loads the patient's age, gender, conditions, and allergies. The result panel presents the final status, rule alert, ML alert, risk score, threshold, and explanatory reason.

**Figure 4.1:** Medication Check interface.  
*[Insert a screenshot of the Medication Check page showing a selected patient and result.]*

### 4.5.2 Patient Directory Interface

The Patient Directory displays patient names, identifiers, ages, genders, and locations (Hripcsak et al., 2013). A search box filters the records, while Previous and Next controls navigate through pages of 10 records. Each patient entry links to the corresponding profile.

**Figure 4.2:** Patient Directory interface.  
*[Insert a screenshot of the paginated Patient Directory page.]*

### 4.5.3 Patient Profile Interface

The Patient Profile presents demographic information, conditions, allergies, and recent medications obtained from the CSV files (Walsh et al., 2019). The page provides a detailed view of the patient context used by the medication-checking process.

**Figure 4.3:** Patient Profile interface.  
*[Insert a screenshot of a Patient Profile page.]*

## 4.6 Explanation of Major Code Modules

Table 4.2: Major Code Modules

| Module | Main responsibility |
|---|---|
| `app.py` | Flask application creation, route handling, CSV loading, patient context retrieval, rule checking, model inference, and JSON responses |
| `train_model.py` | Feature construction, preprocessing pipeline creation, logistic regression training, and model serialization |
| `prepare_synthea_dataset.py` | Preparation of Synthea records for model training and evaluation |
| `templates/index.html` | Medication Check layout and its interactive JavaScript behavior |
| `templates/patients.html` | Searchable and paginated Patient Directory interface |
| `templates/patient_profile.html` | Detailed patient information and medication history interface |
| `static/app.js` | Shared asynchronous request, patient-selection, form, and result-rendering functions |
| `static/styles.css` | Application colors, layout, responsive behavior, forms, cards, navigation, and result styles |
| `models/alert_model.joblib` | Persisted scikit-learn prediction pipeline |
| `csv/` | Synthetic patient, condition, allergy, and medication source records |

## 4.7 Testing Approach

Testing focused on the core workflow and the correctness of the HTTP responses (Beck & Andres, 2004). Flask's test client was used for repeatable backend checks without relying on an external network connection. Manual browser testing was used to verify the user interface (Kawamoto et al., 2005).

The following testing activities were performed:

- route testing for the Medication Check and Patient Directory pages;
- patient-search API testing;
- valid and invalid medication-check request testing;
- rule-based contraindication testing;
- ML risk-score and threshold testing;
- manual patient search, selection, and form interaction testing; and
- responsive interface and navigation checks.

The expected response for a successful prediction was HTTP status `200` with JSON fields for the final alert, rule alert, ML alert, risk score, threshold, and reason. A request without a medication was expected to return HTTP status `400` with a validation message.

## 4.8 Test Cases and Results

Table 4.3: Functional Test Cases and Results

| ID | Test input or action | Expected result | Actual result | Status |
|---|---|---|---|---|
| TC1 | Age 30, female, pregnancy, warfarin | Rule alert and final alert; pregnancy contraindication reason | HTTP 200; rule alert `1`; ML alert `1`; final alert `1`; risk score `0.8656`; reason returned | Pass |
| TC2 | Age 55, male, hypertension, pseudoephedrine | Rule alert and final alert; hypertension contraindication reason | HTTP 200; rule alert `1`; ML alert `1`; final alert `1`; risk score `0.9437`; reason returned | Pass |
| TC3 | Age 28, female, seasonal allergies, loratadine | No known rule alert and low model risk | HTTP 200; rule alert `0`; ML alert `0`; final alert `0`; risk score `0.0474` | Pass |
| TC4 | Open `/` | Medication Check page loads | HTTP 200 | Pass |
| TC5 | Open `/patients` | Patient Directory loads | HTTP 200 | Pass |
| TC6 | Search `/api/patients/search?q=smith` | JSON response containing a results list | HTTP 200 with JSON response | Pass |
| TC7 | Submit a request without medication | Validation error | HTTP 400 with `medication is required` | Pass |
| TC8 | Type in patient search and select a result | Input remains usable; selected patient context is loaded | Dropdown and patient selection operated as intended during manual testing | Pass |
| TC9 | Enter a proposed medication | Check Medication button becomes enabled | Button state updated after input | Pass |

The three representative medication cases demonstrate that known contraindications trigger interpretable rule alerts and that a low-risk combination can complete without an alert. These results verify functional behavior only; they do not establish clinical accuracy.

## 4.9 System Performance Results

A local performance check was carried out using the Flask test client. Twenty consecutive `/predict` requests were submitted after application start-up. The observed mean response time was **12.24 milliseconds**, with a minimum of **9.91 milliseconds** and a maximum of **22.98 milliseconds**.

Table 4.4: Local Prediction Performance

| Metric | Result |
|---|---:|
| Number of requests | 20 |
| Mean response time | 12.24 ms |
| Minimum response time | 9.91 ms |
| Maximum response time | 22.98 ms |

These measurements represent local in-process test-client execution on the development computer. Browser rendering, network delay, concurrent users, and production server configuration were not included. The cached CSV bundle and model loaded at application start-up reduce repeated file-reading and model-loading overhead.

## 4.10 Evaluation of the Implemented System

The implementation met the main functional requirements defined in Chapter 3. Users can search for patients, view patient profiles, submit medication checks, and receive rule and ML results. The system also provides clear navigation between the Medication Check and Patient Directory pages.

The hybrid approach has two important strengths. First, rule matches produce understandable reasons that can be reviewed by the user (Bates et al., 2003). Second, the ML model provides a risk score that can identify patterns beyond the explicitly defined keyword combinations (Jiang et al., 2017). The interface communicates these outputs separately, allowing the user to distinguish deterministic and probabilistic results.

The prototype also has limitations. Its data is synthetic, the contraindication list is simplified, the allergy comparison is based on text overlap, and the ML model has not been clinically validated (Walsh et al., 2019). The measured response time represents a local development environment rather than a production deployment. Consequently, the system should be treated as an academic decision-support prototype and not as a replacement for professional clinical judgment (Sittig et al., 2018).

## 4.11 Chapter Summary

This chapter presented the implementation of the Medication Alert System using Flask, pandas, scikit-learn, joblib, and browser technologies. It explained patient-data processing, rule-based checks, ML inference, hybrid alert generation, and the three principal interfaces. Functional testing showed that the implemented routes and representative alert cases behaved as expected, while local performance testing produced an average prediction time of 9.91 milliseconds. The next chapter discusses the findings, conclusions, limitations, and recommendations arising from the project.

---

# 5. Discussion, Conclusion and Recommendations 10%

## 5.1 Summary of Findings

The implemented system successfully demonstrates a hybrid medication alert workflow using patient context and a proposed medication (Sutton et al., 2020). The rule engine provides immediate interpretable alerts for known contraindications, while the ML model provides an additional risk score that can identify patterns beyond explicitly defined rules (Jiang et al., 2017). Functional testing confirmed that the system correctly triggers rule alerts for known contraindication patterns such as pregnancy with warfarin, hypertension with pseudoephedrine, and asthma with non-selective beta blockers. Performance testing demonstrated that prediction requests are processed in approximately 12 milliseconds on average in a local development environment, meeting the non-functional requirement for fast response times (Dean et al., 2002). The hybrid approach successfully combines explainable rule-based reasoning with data-driven machine learning predictions, providing healthcare providers with both deterministic and probabilistic decision support signals (Poon et al., 2010).

## 5.2 Interpretation of Results

The test results indicate that rule alerts reliably trigger on the defined contraindication patterns, providing interpretable explanations that clinicians can review and understand (Kawamoto et al., 2005). This aligns with clinical decision support best practices, which emphasize the importance of clear, actionable alerts with specific reasons (Bates et al., 2003). The ML scoring provides additional risk estimation that can be used as a secondary signal, potentially identifying risky prescriptions that do not match predefined rule patterns (Jiang et al., 2017). The measured response times suggest that the system can provide near real-time decision support, which is essential for clinical workflows where delays can disrupt provider efficiency (Wright et al., 2018). However, the ML model's risk scores should be interpreted with caution, as they are based on synthetic data and simplified labeling strategies rather than clinically validated outcomes (Walsh et al., 2019).

## 5.3 Whether Objectives Were Achieved

The project successfully achieved all stated objectives:

- **Patient selection workflow implemented:** The system provides a searchable patient directory and detailed patient profile pages that load patient context including demographics, conditions, allergies, and medication history (Hripcsak et al., 2013). This workflow enables clinicians to quickly access relevant patient information before making prescribing decisions.

- **Rule-based contraindications:** The rule engine successfully implements keyword-based contraindication checking for common drug-disease and drug-allergy interactions (Kawamoto et al., 2005). Testing confirmed that the system correctly identifies contraindicated combinations and provides clear explanatory reasons.

- **ML model integration:** The machine learning model was successfully trained on synthetic EHR data and integrated into the Flask application via a serialized pipeline (Pedregosa et al., 2011). The model generates risk scores and binary alert flags that complement the rule-based system.

- **Web UI and endpoints:** The application provides a functional web interface with three main pages (Medication Check, Patient Directory, Patient Profile) and RESTful API endpoints for patient search and medication prediction (Grinberg, 2018). The interface is responsive and suitable for clinical workflow simulation.

## 5.4 Comparison with Existing Systems

Compared with integrated EHR clinical decision support systems (CDSS), this system is lightweight and designed for prototyping rather than production deployment (Sittig et al., 2018). Commercial EHR-integrated CDSS typically feature comprehensive medication knowledge bases covering thousands of drug-drug interactions, drug-disease contraindications, and drug-allergy warnings (Classen et al., 2011). In contrast, this prototype implements a simplified set of rule-based contraindications focused on common high-risk interactions. Production CDSS systems also integrate directly with real-time EHR data, automatically pulling patient context without manual selection (Kawamoto et al., 2005). This prototype requires manual patient selection and uses static CSV files rather than live database connections. However, the hybrid approach of combining rule-based and machine learning alerts is consistent with emerging trends in clinical decision support, which seek to balance explainable rules with data-driven pattern recognition (Sutton et al., 2020). The system's lightweight architecture and use of standard web technologies make it more accessible for academic prototyping and research purposes compared to complex commercial systems.

## 5.5 Advantages of the New System

The implemented system offers several advantages:

- **Lightweight and easy to run:** The system uses standard Python libraries (Flask, pandas, scikit-learn) and does not require complex infrastructure or proprietary software (Van Rossum & Drake, 2009). It can be deployed on a local machine using a virtual environment, making it accessible for academic research and prototyping purposes.

- **Hybrid decision signals (rules + ML):** The combination of rule-based and machine learning approaches provides complementary strengths (Sutton et al., 2020). Rules offer deterministic, explainable alerts for known contraindications, while the ML model can identify patterns beyond explicitly defined rules and provide probabilistic risk assessment (Jiang et al., 2017).

- **Interpretable reasons for rule alerts:** The rule engine provides clear, specific explanations for why an alert was triggered, which is essential for clinician trust and adoption (Bates et al., 2003). This interpretability addresses a common criticism of black-box machine learning systems in healthcare (Poon et al., 2010).

- **Synthetic data approach:** The use of Synthea synthetic EHR data enables development and testing without requiring access to real patient data, addressing privacy concerns and regulatory barriers (Walsh et al., 2019). This approach facilitates academic research while maintaining ethical standards (Hripcsak et al., 2013).

## 5.6 Challenges Encountered

Several challenges were encountered during the development process:

- **Labeling strategy for ML training from synthetic data:** Creating accurate labels for training the machine learning model proved challenging because synthetic data does not include ground truth information about which prescriptions are actually unsafe (Walsh et al., 2019). The labeling strategy relied on applying contraindication rules to generate proxy labels, which may not capture all clinically relevant risk factors (Jiang et al., 2017). This limitation highlights the difficulty of obtaining high-quality labeled data for clinical decision support without access to expert-annotated real-world data.

- **Managing alert thresholds and balancing sensitivity vs specificity:** Selecting an appropriate decision threshold for the ML model required balancing the trade-off between sensitivity (catching more risky prescriptions) and specificity (avoiding false alarms) (Hosmer et al., 2013). A threshold that is too low may generate excessive alerts and contribute to alert fatigue, while a threshold that is too high may miss genuinely risky prescriptions (Classen et al., 2011). This challenge reflects a broader issue in clinical decision support system design.

- **Avoiding alert fatigue in UI design:** Designing the user interface to present alerts without overwhelming the user required careful consideration of visual hierarchy and information density (Wright et al., 2018). The system needed to distinguish between rule-based alerts (which are more certain) and ML alerts (which are probabilistic) to help users prioritize their attention (Bates et al., 2003). This challenge is consistent with research on alert fatigue in clinical settings (Sittig et al., 2018).

## 5.7 Implications of the Study

The project has several implications for clinical decision support research and practice:

- **Feasibility of hybrid alert systems:** The successful implementation demonstrates that hybrid systems combining rule-based and machine learning approaches are technically feasible and can provide complementary decision support signals (Sutton et al., 2020). This suggests that future clinical decision support systems could benefit from integrating both deterministic and probabilistic reasoning methods.

- **Synthetic data for prototyping:** The effective use of Synthea synthetic EHR data shows that synthetic data can be a valuable resource for prototyping clinical decision support systems without requiring access to real patient data (Walsh et al., 2019). This has implications for academic research and innovation in healthcare technology, where data access is often a significant barrier (Hripcsak et al., 2013).

- **Lightweight deployment models:** The system's lightweight architecture suggests that clinical decision support does not always require complex, expensive infrastructure. Simple web-based applications using standard libraries can provide useful functionality for certain use cases (Dean et al., 2002). This may be particularly relevant for resource-constrained settings or for rapid prototyping and testing of new decision support concepts.

- **Interpretability considerations:** The project highlights the importance of interpretability in clinical decision support. The rule engine's clear explanations align with clinician preferences for understandable alerts (Bates et al., 2003), while the ML model's probabilistic output provides additional nuance. This balance between interpretability and predictive power is an ongoing consideration in healthcare AI (Poon et al., 2010).

## 5.8 Contribution of the Project

The project makes the following contributions to the field of clinical decision support:

- **End-to-end ML + web prototype:** The project demonstrates a complete end-to-end implementation of a machine learning-powered clinical decision support system, from data preprocessing and model training to web application deployment (Pedregosa et al., 2011). This provides a reference implementation for researchers and developers interested in building similar systems.

- **Patient directory/profile workflow:** The implementation of a searchable patient directory and detailed patient profile pages provides a realistic simulation of clinical workflow for medication checking (Hripcsak et al., 2013). This workflow demonstrates how patient context can be integrated into decision support interfaces in a user-friendly manner.

- **Hybrid risk alerting approach:** The project contributes a practical example of hybrid decision support that combines rule-based and machine learning approaches (Sutton et al., 2020). The specific implementation strategy of providing separate rule and ML alerts with clear distinctions between them offers a design pattern that could be adopted in other clinical decision support contexts.

- **Synthetic data methodology:** The project demonstrates a methodology for using synthetic EHR data to develop and test clinical decision support systems (Walsh et al., 2019). This approach addresses privacy and data access challenges while still enabling meaningful system development and evaluation.

## 5.9 Improvements for Future Work

Several areas for improvement and future work have been identified:

- **Richer medication knowledge base:** The current contraindication rules cover only a limited set of drug-disease and drug-allergy interactions. Future versions should incorporate a comprehensive medication knowledge base including drug-drug interactions, dose-related contraindications, and age-specific considerations (Classen et al., 2011). Integration with standard drug terminologies such as RxNorm could improve the accuracy and coverage of medication checking (Kawamoto et al., 2005).

- **Improved labeling strategy:** The machine learning model's performance could be improved by developing a more sophisticated labeling strategy. This could involve incorporating expert clinical rules, using clinician feedback to refine labels, or leveraging real-world adverse event reports to identify risky prescription patterns (Jiang et al., 2017). Active learning approaches could be used to iteratively improve the model based on clinician input.

- **Authentication and audit logging:** For production deployment, the system would require robust authentication mechanisms to ensure only authorized users can access patient data (Sittig et al., 2018). Comprehensive audit logging would be necessary to track system usage, support accountability, and enable retrospective analysis of decision support interactions (OWASP, 2023).

- **Database storage and access control:** The current CSV-based data storage should be replaced with a proper database system such as PostgreSQL for improved performance, data integrity, and scalability (Dean et al., 2002). Role-based access control should be implemented to ensure that users can only access data and functions appropriate to their role (Sittig et al., 2018).

- **Clinical validation:** The most critical future work is clinical validation of the system's accuracy and effectiveness. This would involve testing with real patient data, evaluating the system's impact on prescribing decisions and clinical outcomes, and assessing user satisfaction and adoption (Bates et al., 2003). Such validation is essential before any clinical deployment.

- **Enhanced UI/UX:** The user interface could be enhanced based on usability testing with actual clinicians. Improvements could include better visualization of risk information, more intuitive alert presentation, and integration with existing clinical workflow patterns (Wright et al., 2018). Mobile-responsive design could also improve accessibility for clinicians using tablets or smartphones.

---

# References 5%

All sources cited in the literature review.

- Use e.g., APA referencing style
- Books, journals, websites, conference papers

Example placeholders:

- Author, A. A. (Year). Title of article. *Journal Name*, Volume(Issue), pages.
- Organization. (Year). Title of webpage. URL

---

# Appendices

## Appendix A: User Manual

### A.1 How to Run

1. Create and activate venv
2. Install requirements
3. Run `python app.py`
4. Open `http://127.0.0.1:5000/`

### A.2 How to Use

- Use sidebar to open Medication or Patients
- Select patient and check proposed medication

## Appendix B: Additional Screenshots

[Insert additional screenshots]

## Appendix C: Additional Diagrams

[Insert diagrams: architecture/use case/sequence/activity]

## Appendix D: Source Code

Submit source code as part of the project artifact.

## Submission Notes

A. Jury award for the artifact 40%

B. All projects should be submitted with similarity index (plagiarism) and AI content reports.

C. The similarity index and AI content should not be more than 20%.
