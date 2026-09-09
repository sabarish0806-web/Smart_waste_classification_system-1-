# Smart Waste Classification System (EcoSort AI)

A modern, AI-powered web application and computer vision classification system designed to automate waste segregation, educate users on recycling practices, and log classification analytics.

Built according to **Product Requirements Document (PRD) v1.0** specifications.

---

## Key Features

- ♻️ **Automated CV Classification**: Classifies waste images into standard core categories: **Plastic**, **Paper**, **Metal**, **Glass**, **Organic**, and **Other/Unknown**.
- 📊 **Confidence Scoring & Low-Confidence Warnings**: Displays prediction certainty percentages with clear visual meter bars. Triggers automatic re-upload warning alerts when confidence falls below 60% (FR-6).
- 🚮 **Actionable Disposal Guidance**: Recommends bin type, bin color coding, step-by-step handling rules, and common category item examples (FR-5).
- 🖱️ **Drag-and-Drop Image Uploader**: Responsive web UI supporting drag-and-drop or file browser uploads for JPG and PNG images up to 10MB (FR-1, FR-9).
- 🕒 **Session History**: Persists and displays recent scanned items in the current user session.
- 📈 **Admin Analytics Dashboard**: Real-time aggregated statistics showing total scans logged, average confidence metrics, low-confidence warning rates, and category distribution breakdowns (FR-8).
- ⚡ **High Performance**: Sub-second image feature extraction and model inference (NFR Performance KPI: <3 seconds).

---

## System Architecture Overview

```
                          ┌────────────────────────┐
                          │   Frontend Web UI      │
                          │ (HTML5, Tailwind, JS)  │
                          └───────────┬────────────┘
                                      │ Upload / REST API
                                      ▼
                          ┌────────────────────────┐
                          │   Flask REST API Server│
                          │       (app.py)         │
                          └─────┬────────────┬─────┘
                                │            │
                                ▼            ▼
┌─────────────────────────────────┐   ┌───────────────────────────────┐
│     PyTorch & CV ML Engine      │   │     SQLite Database           │
│   (ml_engine.py / torchvision)  │   │  (Flask-SQLAlchemy models.py) │
└─────────────────────────────────┘   └───────────────────────────────┘
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- `pip` package manager

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application Server
```bash
python app.py
```
The server will initialize the SQLite database and start on `http://127.0.0.1:5000`.

---

## Running Automated Tests

Run the complete Pytest suite (ML engine unit tests & Flask REST API integration tests):

```bash
python -m pytest tests/
```

---

## PRD Requirement Traceability Matrix

| Requirement ID | Description | Implementation Status |
|---|---|---|
| **FR-1** | Image upload (JPG, PNG) up to 10MB | ✅ Implemented in `app.py`, `utils.py`, `app.js` |
| **FR-2** | ML/CV image processing | ✅ Implemented in `ml_engine.py` using PyTorch & OpenCV |
| **FR-3** | Classify into waste categories | ✅ Plastic, Paper, Metal, Glass, Organic, Other/Unknown |
| **FR-4** | Display prediction & confidence score | ✅ Dynamic UI badge, progress bar, & percentage text |
| **FR-5** | Display disposal & recycling guidance | ✅ Bin type, color code, handling rules, examples |
| **FR-6** | Low-confidence warning alert (<60%) | ✅ Implemented notification banner with retry recommendation |
| **FR-7** | Database logging of classification results | ✅ Implemented SQLite logging via `ClassificationLog` |
| **FR-8** | Admin analytics dashboard | ✅ Implemented `/api/analytics` & Admin UI tab |
| **FR-9** | Drag-and-drop image upload | ✅ Implemented interactive drag zone in `index.html` & `app.js` |

---

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0 (AGPL-3.0).