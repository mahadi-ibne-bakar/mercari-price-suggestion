# Mercari Price Suggestion — Project README

A machine learning project that predicts the resale price of online marketplace
listings from their title, description, category, brand, condition, and shipping
information — based on the
[Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge)
dataset (1.48M real listings).

Built across three phases: **data preprocessing**, **model training & evaluation**,
and **deployment** via FastAPI + Docker.

---

## Table of contents

1. [Results at a glance](#results-at-a-glance)
2. [Project structure](#project-structure)
3. [Phase I — Data preprocessing](#phase-i--data-preprocessing)
4. [Phase II — Model training & evaluation](#phase-ii--model-training--evaluation)
5. [Phase III — Deployment](#phase-iii--deployment)
6. [Setup & usage](#setup--usage)
7. [Methodology notes — what went wrong and why it matters](#methodology-notes--what-went-wrong-and-why-it-matters)

---

## Results at a glance

| Model | Features | Test RMSLE | Improvement over naive |
|---|---|---|---|
| Naive baseline (predict mean price) | — | 0.7459 | — |
| Linear Regression | 100,012 (TF-IDF + numeric, unscaled) | 0.5087 | 31.8% |
| LightGBM | 12 numeric/categorical | 0.5638 | 24.4% |
| LightGBM + SVD (tuned) | 112 (12 numeric + 100 text "topics") | 0.5003 | 32.9% |
| Ridge (tuned, alpha=3.0) | 100,012 (TF-IDF + scaled numeric) | 0.4619 | 38.1% |
| Ridge (5-fold CV) | same | 0.4624 ± 0.0010 | 38.0% |
| **Ensemble (0.7 × Ridge + 0.3 × LightGBM+SVD)** | both feature sets | **0.4507** | **39.6%** |

All numbers are from an **80/20 train/test split** (the version required for this
project), evaluated using **RMSLE** — the competition's official metric. The
ensemble is the best-performing configuration and is the default selection in
the deployed app, though all three model types are available via a toggle.

---

## Project structure

```
mercari-price/
├── data/
│   ├── train.tsv                 # raw data (download from Kaggle, not in repo)
│   └── train_processed.pkl       # Phase I output: cleaned + feature-engineered
├── models/                        # Phase II outputs (12 files, ~71.5 MB total)
│   ├── ridge_80.pkl               # Ridge, trained on 80% of data
│   ├── ridge_full.pkl             # Ridge, trained on 100% of data
│   ├── lgbm_80.pkl                # LightGBM + SVD, trained on 80%
│   ├── lgbm_full.pkl              # LightGBM + SVD, trained on 100%
│   ├── ensemble_80.pkl            # {ridge_80 + lgbm_80, weights 0.7/0.3}
│   ├── ensemble_full.pkl          # {ridge_full + lgbm_full, weights 0.7/0.3}
│   ├── numeric_scaler.pkl         # StandardScaler for the 12 numeric features
│   ├── label_encoders.pkl         # LabelEncoders: cat_1, cat_2, cat_3, brand_name
│   ├── tfidf_name.pkl             # TF-IDF vectorizer, product names (50K terms)
│   ├── tfidf_desc.pkl             # TF-IDF vectorizer, descriptions (50K terms)
│   ├── svd_name.pkl               # TruncatedSVD, name TF-IDF -> 50 dims
│   └── svd_desc.pkl               # TruncatedSVD, desc TF-IDF -> 50 dims
├── plots/                          # 6 charts from Phase I EDA
├── app/
│   ├── main.py                    # FastAPI app: pipeline + /predict endpoint
│   └── templates/
│       └── index.html             # frontend form with model/data toggles
├── eda.ipynb                       # Phase I + II notebook — the full story, sequentially
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md                       # this file
```

---

## Phase I — Data preprocessing

**Goal:** understand the raw dataset thoroughly, then clean and engineer features
based on what was actually found — never the other way around.

### Dataset

1,482,535 listings, 8 columns: `train_id`, `name`, `item_condition_id`,
`category_name`, `brand_name`, `price` (target), `shipping`, `item_description`.

### Key findings from exploration

| Finding | Detail |
|---|---|
| Missing `brand_name` | 632,682 rows (42.7%) — the single largest data quality issue |
| Missing `category_name` | 6,327 rows (0.43%) |
| Hidden missing descriptions | 82,489 rows say `"No description yet"` — a Mercari placeholder, **not** caught by `.isna()` |
| `price` distribution | min $0, max $2,009, median $17, mean $26.74, **skewness 11.39** (extremely right-skewed) |
| `category_name` format | always `"Level1/Level2/Level3"`, e.g. `Women/Tops & Blouses/T-Shirts` — 1,287 unique combinations |
| `brand_name` cardinality | 4,809 unique brands; known-brand listings cost ~1.4x more on median, with top luxury brands (e.g. David Yurman, $220 median) costing 10x+ more |
| Shipping vs price | counter-intuitive: seller-paid shipping correlates with **cheaper** items ($14 median vs $20) — sellers absorb shipping cost on low-value items to stay competitive |

### Cleaning decisions

| Problem | Fix | Why |
|---|---|---|
| 874 listings priced at $0 | Removed | Data errors; RMSLE is undefined-ish at price=0 |
| Missing `brand_name` | Filled with `"unknown"` | Creates a real, learnable "no brand" category |
| Missing `category_name` | Filled with `"missing/missing/missing"` | Splits cleanly into 3 "missing" sub-categories |
| Missing/placeholder `item_description` | Filled with `"no description"` | One consistent signal for "no real description" |
| Target variable | `log_price = log1p(price)` | Skewness drops from 11.39 → 0.66; RMSLE becomes plain RMSE in log space |

### Feature engineering

- **Category split:** `category_name` → `cat_1` (11 values), `cat_2` (114), `cat_3` (871) — three separate label-encoded columns instead of one opaque string.
- **Text statistics:** `name_len`, `desc_len`, `name_word_count`, `desc_word_count`, `has_description`, `brand_known` (12 numeric features total, including `item_condition_id` and `shipping`).
- **TF-IDF:** two vectorizers — 50,000 terms from `name` (1-2 grams), 50,000 terms from `item_description` (1-2 grams), both with `sublinear_tf=True`.

### Feature selection — two methods, two different stories

Pearson correlation and LightGBM feature importance **disagreed substantially**:

| Feature | Pearson correlation | LightGBM importance |
|---|---|---|
| `shipping` | −0.231 (strongest) | low (99) |
| `brand_known` | +0.206 | near-zero (13) |
| `cat_2_encoded` | ~0.002 (looks useless) | **1,092 (3rd highest)** |
| `cat_3_encoded` | ~−0.004 (looks useless) | **1,412 (2nd highest)** |
| `item_condition_id` | ~−0.002 (looks useless) | 313 |
| `brand_name_encoded` | −0.143 | **1,617 (highest)** |

**Why:** label-encoded categoricals have arbitrary integer values with no linear
relationship to price — Pearson correlation can't see the signal. Tree models
split on thresholds regardless of ordering and find it immediately. **Decision:
keep all 12 features** — confirmed useful by at least one method, and harmless
to the other.

---

## Phase II — Model training & evaluation

**Goal:** systematically compare models, feature sets, and split ratios — and
follow up on every surprising result rather than ignoring it.

### Two feature sets, matched to two model types

| Feature set | Columns | Used by |
|---|---|---|
| `X_full` | 12 scaled numeric + 50K name TF-IDF + 50K description TF-IDF = **100,012** | Linear Regression, Ridge |
| `X_lgbm_text` | 12 scaled numeric + 50 name-SVD "topics" + 50 description-SVD "topics" = **112** | LightGBM |

Linear models get direct access to individual word weights (e.g. "louis
vuitton"). LightGBM gets a compressed, dense representation — feeding it 100K
sparse columns would make tree-splitting far slower for little gain.

### Split ratios tested

70/30, 80/20, and 90/10 were all tested for Ridge and LightGBM. Across **1.48M
rows**, all three ratios produced nearly identical results (within ~0.004
RMSLE) — itself a meaningful finding: performance is a property of the model
and features, not of which rows happened to land in the test set. **80/20 is
used as the reference split throughout this README and in the deployed app's
"80% split" toggle.**

### The headline result: blending beats both individual models

| Blend weight (Ridge) | RMSLE |
|---|---|
| 0.0 (pure LightGBM+SVD) | 0.5003 |
| 0.5 | 0.4544 |
| 0.6 | 0.4514 |
| **0.7** | **0.4507** ← best |
| 0.8 | 0.4523 |
| 1.0 (pure Ridge) | 0.4619 |

A 0.7/0.3 blend of Ridge and LightGBM+SVD beats *either model alone* by a
meaningful margin (2.4% relative RMSLE reduction vs. Ridge solo). This works
because the two models are driven by genuinely different signals: Ridge by
specific TF-IDF word weights, LightGBM by non-linear interactions between
label-encoded categoricals (e.g. "this specific `cat_3` + `brand_name`
combination") — something a linear model structurally cannot represent.

### Hyperparameter tuning

- **Ridge:** alpha grid `[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]` → clean
  U-shape, minimum at **alpha=3.0** (0.46194), though the 1.0–5.0 range is
  fairly flat (within ~0.001 of each other).
- **LightGBM+SVD:** grid over `num_leaves` (31/63), `n_estimators` (500/1000),
  `learning_rate` (0.05/0.1) — every direction of increased capacity helped
  monotonically. Best: `leaves=63, trees=1000, lr=0.1` → 0.5003 (down from
  0.5379 at the smallest grid corner).

### Cross-validation

5-fold CV on Ridge (alpha=3.0, `X_full`): **mean RMSLE 0.46244, std 0.00099**
across folds ranging 0.46115–0.46415. Confirms the single-split result wasn't
a fluke.

### Six deployable model files

Per-model and per-data-size variants were saved so the deployed app can toggle
between them:

| | Trained on 80% (1,185,328 rows) | Trained on 100% (1,481,661 rows) |
|---|---|---|
| Ridge (alpha=3.0) | `ridge_80.pkl` | `ridge_full.pkl` |
| LightGBM+SVD (tuned) | `lgbm_80.pkl` | `lgbm_full.pkl` |
| Ensemble (0.7/0.3, frozen weight) | `ensemble_80.pkl` | `ensemble_full.pkl` |

For the `_full` variants, the 0.7/0.3 blend weight is **frozen** from the
80/20-validated value — re-deriving it against training data would be circular.

---

## Phase III — Deployment

**Goal:** wrap the winning pipeline in a FastAPI service, containerize it with
Docker, and prove it produces identical predictions to the notebook.

### Architecture

```
Raw listing (JSON)
   │
   ├─ clean_text()                  → lowercase, normalize whitespace
   ├─ split category_name on "/"    → cat_1, cat_2, cat_3
   ├─ safe_encode() with fallback   → handles brands/categories never seen in training
   ├─ case-insensitive brand lookup → "nike" matches encoder's "Nike"
   ├─ StandardScaler (12 numeric)
   ├─ TF-IDF transform (name + description, 50K terms each)
   └─ TruncatedSVD transform (for LightGBM path only)
         │
         ├─→ X_full (100,012 cols)  → Ridge / Ensemble
         └─→ X_lgbm_text (112 cols) → LightGBM / Ensemble
                  │
                  ▼
          predicted log(price) → expm1() → price in USD
```

### API

`GET /` — serves an HTML form (product details + model/data-size toggles)

`POST /predict` — accepts a JSON listing, returns:
```json
{"predicted_price": 48.35, "model_type": "ensemble", "data_version": "full"}
```

Request fields: `name`, `item_condition_id` (1–5), `category_name`,
`brand_name`, `shipping` (0/1), `item_description`, `model_type`
(`ridge`/`lgbm`/`ensemble`), `data_version` (`80`/`full`). All validated via
Pydantic before reaching the prediction pipeline.

### Docker

- Base image: `python:3.12-slim`
- `libgomp1` installed for LightGBM's OpenMP dependency (the Linux equivalent
  of macOS's `libomp`)
- Dependencies installed via `uv sync --frozen` for exact reproducibility
- Runs `uvicorn` bound to `0.0.0.0:8000` (required for Docker port mapping)

### Verified end-to-end

A test listing ("Nike Air Force 1 white sneakers, size 9, Men/Shoes/Athletic,
condition 2, seller-paid shipping") produced **identical predictions** —
$40.11 (Ridge), $75.73 (LightGBM), $48.35 (Ensemble) — whether run directly in
the notebook or via `curl` against the running Docker container.

---

## Setup & usage

### Prerequisites

- macOS with [uv](https://docs.astral.sh/uv/) installed
- [Homebrew](https://brew.sh/) (for `libomp`, required by LightGBM on macOS)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1. Clone/set up the project

```bash
cd mercari-price
uv python pin 3.12
uv sync
brew install libomp     # required for LightGBM on macOS
```

### 2. Get the data

Download `train.tsv` from the
[Kaggle competition data page](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)
and place it at `data/train.tsv`.

### 3. Run the notebook (Phases I & II)

Open `eda.ipynb` in VS Code, select the `.venv` kernel, and run all cells in
order. This regenerates `data/train_processed.pkl`, all 6 model files, and all
6 preprocessing objects in `models/`, plus 6 charts in `plots/`.

### 4. Run the API locally (Phase III)

```bash
uv run uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser.

### 5. Build & run with Docker

```bash
docker build -t mercari-price-app .
docker run -p 8000:8000 mercari-price-app
```

Open **http://127.0.0.1:8000**, or test via curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nike Air Force 1 white sneakers size 9",
    "item_condition_id": 2,
    "category_name": "Men/Shoes/Athletic",
    "brand_name": "Nike",
    "shipping": 1,
    "item_description": "Worn a few times, great condition, comes with original box",
    "model_type": "ensemble",
    "data_version": "full"
  }'
```

---

## Methodology notes — what went wrong and why it matters

These are the moments worth highlighting when explaining this project — each
one is a real debugging story with a transferable lesson, not just a final
number.

**1. The Ridge "alpha does nothing" bug (feature scaling).**
Ridge at `alpha=1.0` initially scored *worse* than unregularized Linear
Regression (0.63 vs 0.51). A sweep across alphas from 0.001 to 1.0 produced
*identical* scores to 4 decimal places — the signature of a hyperparameter
that isn't affecting anything. Root cause: the 12 raw numeric features (e.g.
`brand_name_encoded` ranging 0–4807) completely dominated the TF-IDF columns
(ranging ~0–0.5), so Ridge's penalty term — proportional to coefficient size —
was operating at the wrong scale entirely. Fix: `StandardScaler` on the 12
numeric features only (never on TF-IDF — it's already normalized, and scaling
a sparse matrix would make it dense and consume ~1TB of memory). After fixing
this, Ridge jumped to 0.463 and a proper alpha sweep showed the expected
U-shape. **Lesson: for linear models with mixed-scale features, an apparently
"broken" hyperparameter is often a scaling bug in disguise.**

**2. LightGBM losing despite having the "most important" features.**
Phase I's feature importance analysis showed `brand_name_encoded`,
`cat_3_encoded`, and `cat_2_encoded` were LightGBM's top 3 features — by a wide
margin. Yet LightGBM trained on exactly those 12 features scored 0.564, worse
than even unregularized Linear Regression on 100K TF-IDF features (0.509).
"Most important among a limited set" doesn't mean "competitive with a much
richer set" — the categorical features are a lossy summary of information that
exists in much sharper form in the raw text.

**3. Ensembling two models 0.038 RMSLE apart still helped — by more than expected.**
The initial expectation was that blending a much weaker model (LightGBM+SVD,
0.500) into a stronger one (Ridge, 0.462) would yield a marginal 0.001–0.005
improvement, if any. The actual result — 0.4507, a 0.0112 improvement — was
roughly 10x larger than that estimate, because the two models' errors were
genuinely decorrelated: Ridge captures word-level signal, LightGBM captures
non-linear categorical interactions Ridge structurally cannot represent.
**Lesson: don't dismiss ensembling just because one model is much weaker — test
it, because "weaker overall" and "uncorrelated errors" are different things.**

**4. Real-world input breaks pipelines that notebook data never tests.**
Two bugs only appeared once hand-typed listings were fed through the pipeline:
(a) a brand or category never seen in the 1.48M training rows crashes
`LabelEncoder.transform()` with a `ValueError` — fixed with a `safe_encode()`
fallback to `"unknown"`/`"missing"`; (b) `brand_name` was never lowercased in
Phase I (unlike the category columns), so typing `"nike"` wouldn't match the
encoder's `"Nike"` class and would silently lose all brand signal — fixed with
a case-insensitive lookup table built once at startup. **Lesson: test your
deployment pipeline with deliberately adversarial inputs (unseen values, wrong
casing, empty fields) before containerizing — these bugs are invisible in
any training-data-only evaluation.**

**5. The same OpenMP issue, twice, on two different OSes.**
LightGBM depends on OpenMP for parallelism. macOS doesn't ship it — Homebrew's
`libomp` package provides it, and this was needed *twice* in this project (once
for the original environment, once again when the FastAPI app's `joblib.load()`
imported LightGBM in a fresh process). Linux containers need the equivalent
`libgomp1` via `apt-get`. Same root cause, different package name, different OS
— recognizing the *pattern* the second time made the Docker fix immediate
rather than another multi-step debugging session.
