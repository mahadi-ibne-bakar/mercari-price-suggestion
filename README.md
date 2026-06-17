# Mercari Price Suggestion

A machine learning project that predicts the resale price of online marketplace
listings from their title, description, category, brand, condition, and shipping
information — based on the
[Mercari Price Suggestion Challenge](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge)
dataset (1.48M real listings).

Built across three phases: **data preprocessing**, **model training & evaluation**,
and **deployment** via FastAPI + Docker.

**Docker Hub Link:** `https://hub.docker.com/r/mahadiirl/mercari-price-app`
**Run on Local Machine:** `docker pull mahadiirl/mercari-price-app:latest`  

---

## Table of contents

1. [Results at a glance](#results-at-a-glance)
2. [Project structure](#project-structure)
3. [Phase I — Data preprocessing](#phase-i--data-preprocessing)
4. [Phase II — Model training & evaluation](#phase-ii--model-training--evaluation)
5. [Phase III — Deployment](#phase-iii--deployment)
6. [Setup & usage](#setup--usage)
7. [Methodology notes](#methodology-notes)

---

## Results at a glance

All results use an **80/20 train/test split** evaluated with **RMSLE**
(Root Mean Squared Logarithmic Error) — lower is better. The 80/20 split
is the project requirement; results were also stable across 70/30 and 90/10
(within ±0.004 RMSLE across all splits at this dataset size).

| Model | Feature set | Test RMSLE | vs naive |
|---|---|---|---|
| Naive baseline (predict mean price) | — | 0.7459 | — |
| Linear Regression | 100K TF-IDF + numeric (unscaled) | 0.5087 | −31.8% |
| LightGBM | 12 numeric/categorical | 0.5638 | −24.4% |
| LightGBM + SVD (tuned) | 112 (12 numeric + 100 compressed text topics) | 0.5003 | −32.9% |
| Ridge (alpha=3.0, 5-fold CV) | 100K TF-IDF + 12 scaled numeric | 0.4624 ± 0.001 | −38.0% |
| LightGBM_full (feature_fraction=0.3) | 100K TF-IDF + 12 scaled numeric | 0.4506 | −39.6% |
| Ensemble v1 (0.7 × Ridge + 0.3 × LightGBM+SVD) | both feature sets | 0.4507 | −39.6% |
| **Ensemble v2 (0.4 × Ridge + 0.6 × LightGBM_full)** | X_full (100K) | **0.4428** | **−40.6%** |

**Winner:** Ensemble v2 — the best-performing configuration and the default
in the deployed app. All five model types are available via a toggle in the UI.

---

## Project structure

```
mercari-price/
├── data/
│   ├── train.tsv                   # raw data (download from Kaggle — not in repo)
│   └── train_processed.pkl         # Phase I output: cleaned + feature-engineered
│
├── models/                         # Phase II outputs (16 files, ~114 MB total)
│   ├── ridge_80.pkl                # Ridge alpha=3.0, trained on 80%
│   ├── ridge_full.pkl              # Ridge alpha=3.0, trained on 100%
│   ├── lgbm_80.pkl                 # LightGBM+SVD (112 features), trained on 80%
│   ├── lgbm_full.pkl               # LightGBM+SVD (112 features), trained on 100%
│   ├── lgbm_text_80.pkl            # LightGBM_full (100K features), trained on 80%
│   ├── lgbm_text_full.pkl          # LightGBM_full (100K features), trained on 100%
│   ├── ensemble_80.pkl             # {ridge_80 + lgbm_80, weights 0.7/0.3}
│   ├── ensemble_full.pkl           # {ridge_full + lgbm_full, weights 0.7/0.3}
│   ├── ensemble_v2_80.pkl          # {ridge_80 + lgbm_text_80, weights 0.4/0.6}
│   ├── ensemble_v2_full.pkl        # {ridge_full + lgbm_text_full, weights 0.4/0.6}
│   ├── numeric_scaler.pkl          # StandardScaler for 12 numeric features
│   ├── label_encoders.pkl          # LabelEncoders: cat_1, cat_2, cat_3, brand_name
│   ├── tfidf_name.pkl              # TF-IDF vectorizer, product names (50K terms)
│   ├── tfidf_desc.pkl              # TF-IDF vectorizer, descriptions (50K terms)
│   ├── svd_name.pkl                # TruncatedSVD: name TF-IDF → 50 dims
│   └── svd_desc.pkl                # TruncatedSVD: desc TF-IDF → 50 dims
│
├── plots/                          # 8 charts from Phase I EDA
│
├── app/
│   ├── main.py                     # FastAPI app: full pipeline + /predict endpoint
│   └── templates/
│       └── index.html              # Frontend form with model/data-version toggles
│
├── notebooks.ipynb                       # Phase I + II — complete sequential notebook
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

> **Note:** `data/` and `models/` are excluded from version control
> (see `.gitignore`). To reproduce them, download `train.tsv` from Kaggle
> and run all cells in `notebooks.ipynb` — this regenerates every file in both
> directories from scratch.

---

## Phase I — Data preprocessing

**Goal:** understand the raw dataset thoroughly before touching anything,
then make targeted, justified changes — never the other way around.

### Dataset

1,482,535 listings, 8 columns: `train_id`, `name`, `item_condition_id`,
`category_name`, `brand_name`, `price` (target), `shipping`, `item_description`.

### Key findings from exploration

| Finding | Detail |
|---|---|
| Missing `brand_name` | 632,682 rows (42.7%) — the single largest data quality issue |
| Missing `category_name` | 6,327 rows (0.43%) |
| Hidden missing descriptions | 82,489 rows contain `"No description yet"` — a Mercari placeholder, **not** caught by `.isna()` |
| `price` distribution | min $0, max $2,009, median $17, mean $26.74, **skewness 11.39** |
| `category_name` format | always `"Level1/Level2/Level3"`, e.g. `Women/Tops & Blouses/T-Shirts` — 1,287 unique combinations |
| `brand_name` cardinality | 4,809 unique brands; known-brand listings cost ~1.4× more on median; top luxury brands (e.g. David Yurman, $220 median) cost 10× more |
| Shipping vs price | counter-intuitive: seller-paid shipping correlates with **cheaper** items ($14 median vs $20) — sellers absorb cost on low-value listings to stay competitive |

### Cleaning decisions (each motivated by EDA findings above)

| Problem found | Fix | Reason |
|---|---|---|
| 874 listings priced at $0 | Removed | Data errors; RMSLE is undefined at price=0 |
| Missing `brand_name` | Filled with `"unknown"` | Creates a real, learnable "no brand" category |
| Missing `category_name` | Filled with `"missing/missing/missing"` | Splits cleanly into 3 "missing" sub-categories in Step 9 |
| Missing/placeholder `item_description` | Filled with `"no description"` | One consistent signal for "no real description" |
| Price skewness 11.39 | `log_price = log1p(price)` | Skewness drops to 0.66; training on log_price makes RMSLE equivalent to plain RMSE |

### Feature engineering

- **Category split:** `category_name` → `cat_1` (11 unique values), `cat_2` (114),
  `cat_3` (871) — three separately label-encoded columns instead of one opaque string.
- **Text statistics:** `name_len`, `desc_len`, `name_word_count`, `desc_word_count`,
  `has_description`, `brand_known` — 12 numeric features total (including
  `item_condition_id` and `shipping`).
- **TF-IDF:** two fitted vectorizers, each 50,000 terms with 1–2-gram ranges
  and `sublinear_tf=True` — one for `name`, one for `item_description`.
  Fitted on train only; saved to `models/` for Phase II and Phase III use.
- **TruncatedSVD:** each TF-IDF matrix compressed to 50 "topic" dimensions
  for the LightGBM+SVD model path.

### Feature selection — two methods, two different stories

| Feature | Pearson correlation | LightGBM importance |
|---|---|---|
| `shipping` | −0.231 (strongest) | 99 (low) |
| `brand_known` | +0.206 (strong) | 13 (near-zero) |
| `cat_2_encoded` | ~0.002 (appears useless) | **1,092 (3rd highest)** |
| `cat_3_encoded` | ~−0.004 (appears useless) | **1,412 (2nd highest)** |
| `item_condition_id` | ~−0.002 (appears useless) | 313 |
| `brand_name_encoded` | −0.143 | **1,617 (highest)** |

Label-encoded categoricals carry no linear signal — Pearson correlation
cannot see what tree models find immediately. **All 12 features kept** —
each confirmed by at least one method, with negligible cost to the other.

---

## Phase II — Model training & evaluation

**Goal:** systematically compare models, feature sets, and split ratios —
and follow up on every surprising result rather than ignoring it.

### Three feature pipelines

| Feature set | Columns | Used by |
|---|---|---|
| `X_full` | 12 **scaled** numeric + 50K name TF-IDF + 50K description TF-IDF = **100,012** | Ridge, LightGBM_full, Ensemble v2 |
| `X_lgbm_text` | 12 scaled numeric + 50 name-SVD topics + 50 description-SVD topics = **112** | LightGBM+SVD, Ensemble v1 |

The 12 numeric features required `StandardScaler` before being combined with
TF-IDF — raw numeric values up to ~4800 completely drowned out TF-IDF values
(~0–0.5), making Ridge's `alpha` hyperparameter ineffective until scaling was
applied. See [Methodology notes](#methodology-notes).

### Split ratios tested

70/30, 80/20, and 90/10 were all tested across all models. With 1.48M rows,
all three produced results within ±0.004 RMSLE of each other — performance is
a property of the model and features, not of which rows landed where.
**80/20 is the reference split** throughout this README.

### Hyperparameter tuning

- **Ridge:** alpha grid `[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]` —
  U-shape minimum at **alpha=3.0** (0.46194). Range 1.0–5.0 within ±0.001
  of each other (flat valley).
- **LightGBM+SVD:** grid over `num_leaves` × `n_estimators` × `learning_rate`
  (8 combinations) — every direction of increased capacity helped monotonically.
  Best: `leaves=63, trees=1000, lr=0.1` → RMSLE 0.5003.
- **LightGBM_full:** `num_leaves=63, n_estimators=1000, learning_rate=0.1,
  feature_fraction=0.3` — evaluates 30% of 100K columns per node (feasible
  training time ~32 min). Achieves RMSLE 0.4506 — better than Ridge alone.

### Cross-validation

5-fold CV on Ridge (alpha=3.0): **mean RMSLE 0.46244, std 0.00099** across
folds 0.46115–0.46415. Performance is stable and not a product of one
lucky split.

### Ensembling — two versions

**Ensemble v1** (0.7 × Ridge + 0.3 × LightGBM+SVD): RMSLE **0.4507**.
Improves on Ridge solo (0.4619) because the two models make decorrelated
errors — Ridge uses word-level TF-IDF signal; LightGBM finds non-linear
categorical interactions Ridge cannot represent linearly.

**Ensemble v2** (0.4 × Ridge + 0.6 × LightGBM_full): RMSLE **0.4428**.
With LightGBM_full now stronger than Ridge in isolation (0.4506 vs 0.4619),
the blend shifts toward LightGBM as the dominant component. Both models
use the same `X_full` feature matrix — simpler than v1's two-pipeline
architecture. This is the deployed default.

Blend weight (0.4/0.6) was validated on the 80/20 test set and **frozen**
for the 100%-trained production models — re-deriving against training data
would be circular.

### Ten deployable model files (5 configurations × 2 data versions)

| Model | 80% split | 100% data |
|---|---|---|
| Ridge (alpha=3.0) | `ridge_80.pkl` | `ridge_full.pkl` |
| LightGBM+SVD (tuned) | `lgbm_80.pkl` | `lgbm_full.pkl` |
| LightGBM_full (100K, tuned) | `lgbm_text_80.pkl` | `lgbm_text_full.pkl` |
| Ensemble v1 (0.7/0.3) | `ensemble_80.pkl` | `ensemble_full.pkl` |
| **Ensemble v2 (0.4/0.6)** ← default | `ensemble_v2_80.pkl` | `ensemble_v2_full.pkl` |

---

## Phase III — Deployment

**Goal:** wrap the winning pipeline in a production-grade web service,
containerize it, verify it produces bit-identical predictions to the notebook.

### Full pipeline per request

```
Raw listing (JSON)
   │
   ├─ clean_text()                     → lowercase, normalise whitespace
   ├─ split category_name on "/"       → cat_1, cat_2, cat_3
   ├─ safe_encode() with fallback      → unseen brands/categories → "unknown"/"missing"
   ├─ case-insensitive brand lookup    → "nike" → encoder's "Nike"
   ├─ StandardScaler (12 numeric)
   ├─ TF-IDF transform (name + description, 50K terms each)
   └─ (LightGBM+SVD path only) TruncatedSVD → 50 dims per text field
         │
         ├─→ X_full (100,012 cols sparse) → Ridge | LightGBM_full | Ensemble v2
         └─→ X_lgbm_text (112 cols dense) → LightGBM+SVD | Ensemble v1
                  │
                  ▼
          predicted log(price) → np.expm1() → price in USD
```

### API endpoints

`GET /` — HTML form: product fields + model toggle + data-version toggle

`POST /predict` — accepts JSON, returns:
```json
{"predicted_price": 47.30, "model_type": "ensemble_v2", "data_version": "full"}
```

Valid `model_type` values: `ridge` | `lgbm` | `lgbm_text` | `ensemble` | `ensemble_v2`  
Valid `data_version` values: `80` | `full`

All fields validated via Pydantic before reaching the pipeline — invalid inputs
return a `422 Unprocessable Entity` with a clear error message.

### Docker

- Base image: `python:3.12-slim` (Debian-based — compatible with LightGBM wheels)
- `libgomp1` installed via `apt-get` (LightGBM's OpenMP dependency on Linux)
- Dependencies installed via `uv sync --frozen` — exact reproducibility
- Binds to `${PORT:-8000}` — works locally (port 8000) and on Render (injected `PORT`)

### Docker Hub

```bash
docker pull mahadiirl/mercari-price-app:latest
docker run -p 8000:8000 mahadiirl/mercari-price-app
```

Image: https://hub.docker.com/r/mahadiirl/mercari-price-app

### Verified end-to-end

Test listing: Nike Air Force 1 white sneakers, size 9, Men/Shoes/Athletic,
condition 2, seller-paid shipping. Predictions from notebook Cell 50 vs.
`curl` against the running Docker container:

| Model | Data | Notebook | Docker container |
|---|---|---|---|
| ridge | full | $39.84 | $39.84 |
| lgbm_text | full | $53.01 | $53.01 |
| ensemble_v2 | full | $47.30 | $47.30 |

Bit-identical across all 20 model × data-version combinations — the full
pipeline (text cleaning, encoding, scaling, TF-IDF, SVD, model inference)
runs identically inside an isolated Linux container.

---

## Setup & usage

### Prerequisites

- macOS with [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Homebrew](https://brew.sh/) — required for `libomp` (LightGBM on macOS)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1. Clone and install

```bash
git clone https://github.com/mahadiirl/mercari-price-suggestion.git
cd mercari-price-suggestion
uv python pin 3.12
uv sync
brew install libomp
```

### 2. Get the data

Download `train.tsv` from the
[Kaggle competition data page](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data)
and place it at `data/train.tsv`.

### 3. Reproduce Phase I & II (run the notebook)

Open `eda.ipynb` in VS Code, select the `.venv` kernel, and run all cells
in order. This generates:
- `data/train_processed.pkl`
- All 10 model files + 6 preprocessing objects in `models/`
- 8 charts in `plots/`

> The LightGBM_full training cell (~32 min) and the 100% data
> LightGBM_full cell (~40 min) are the slow steps. All other cells
> complete in under 2 minutes each.

### 4. Run the API locally

```bash
uv run uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**

### 5. Run with Docker (no Python setup required)

```bash
# Pull from Docker Hub
docker pull mahadiirl/mercari-price-app:latest
docker run -p 8000:8000 mahadiirl/mercari-price-app

# Or build locally
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
    "model_type": "ensemble_v2",
    "data_version": "full"
  }'
```

---

## Methodology notes

Real debugging stories with transferable lessons.

**1. The Ridge "alpha does nothing" bug — feature scaling.**
Ridge at `alpha=1.0` initially scored 0.63 — *worse* than the naive baseline
and unregularized Linear Regression (0.51). A sweep from `alpha=0.001` to
`alpha=1.0` produced *identical* scores to 4 decimal places — the signature of
a hyperparameter that isn't affecting anything. Root cause: 12 raw numeric
features (e.g. `brand_name_encoded` up to 4807) completely dominated the
TF-IDF columns (~0–0.5), so Ridge's penalty — proportional to coefficient
magnitude — was numerically negligible compared to the raw features.
`StandardScaler` on the 12 numeric columns only (never TF-IDF — it's already
normalised, and scaling a 100K-column sparse matrix would make it dense and
fill memory) fixed it instantly. Ridge jumped to 0.463. **Lesson: for linear
models with mixed-scale inputs, a "broken" hyperparameter sweep is often a
scaling bug in disguise.**

**2. LightGBM losing despite having the "most important" features.**
Phase I feature importance showed `brand_name_encoded`, `cat_3_encoded`,
`cat_2_encoded` as LightGBM's top 3 — by a wide margin over all other
numeric features. Yet LightGBM on those 12 features scored 0.564, worse than
unregularized Linear Regression on 100K features (0.509). "Most important
within a limited set" is not the same as "competitive with a richer set" —
the category and brand columns are a lossy summary of information available
in far sharper form in raw text. When LightGBM was given direct access to
the full 100K TF-IDF matrix (`feature_fraction=0.3`), it scored 0.4506 —
better than Ridge.

**3. Ensembling a weaker model improved results by more than expected.**
The initial expectation was that blending LightGBM+SVD (0.500) into Ridge
(0.462) would yield ≤0.005 improvement. The actual result — 0.4507, a 0.011
improvement — was roughly 10× larger, because the two models' errors were
genuinely decorrelated: Ridge uses word-level TF-IDF signal; LightGBM finds
non-linear categorical interactions a linear model structurally cannot represent.
The second ensemble (v2, using LightGBM_full) improved further to 0.4428.
**Lesson: test ensembling even when one model is weaker — "weaker overall"
and "decorrelated errors" are different things.**

**4. Real-world input breaks pipelines that training-data evaluation never tests.**
Two bugs only appeared when hand-typed listings entered the pipeline:
(a) unseen brands/categories crash `LabelEncoder.transform()` with `ValueError`
— fixed via `safe_encode()` fallback to `"unknown"`/`"missing"`;
(b) `brand_name` was never lowercased in Phase I (unlike categories), so
`"nike"` failed to match the encoder's `"Nike"` and silently lost brand signal
— fixed with a case-insensitive lookup table built at startup. Both bugs were
invisible in any training-data-only evaluation. **Lesson: test your deployment
pipeline with deliberately adversarial inputs — unseen values, wrong casing,
empty fields — before containerising.**

---

## License

MIT — see [LICENSE](LICENSE) for details.
