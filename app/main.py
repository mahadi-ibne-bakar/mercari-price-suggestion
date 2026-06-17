# app/main.py

import re
import warnings
import joblib
import numpy as np
import scipy.sparse as sp
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =====================================================================
# SECTION 1 — STARTUP: load all objects ONCE when the app starts
# =====================================================================

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

print("Loading models and preprocessing objects...")

scaler     = joblib.load(MODELS_DIR / "numeric_scaler.pkl")
encoders   = joblib.load(MODELS_DIR / "label_encoders.pkl")
tfidf_name = joblib.load(MODELS_DIR / "tfidf_name.pkl")
tfidf_desc = joblib.load(MODELS_DIR / "tfidf_desc.pkl")
svd_name   = joblib.load(MODELS_DIR / "svd_name.pkl")
svd_desc   = joblib.load(MODELS_DIR / "svd_desc.pkl")

MODELS = {
    ("ridge",       "80"):   joblib.load(MODELS_DIR / "ridge_80.pkl"),
    ("ridge",       "full"): joblib.load(MODELS_DIR / "ridge_full.pkl"),
    ("lgbm",        "80"):   joblib.load(MODELS_DIR / "lgbm_80.pkl"),
    ("lgbm",        "full"): joblib.load(MODELS_DIR / "lgbm_full.pkl"),
    ("lgbm_text",   "80"):   joblib.load(MODELS_DIR / "lgbm_text_80.pkl"),
    ("lgbm_text",   "full"): joblib.load(MODELS_DIR / "lgbm_text_full.pkl"),
    ("ensemble",    "80"):   joblib.load(MODELS_DIR / "ensemble_80.pkl"),
    ("ensemble",    "full"): joblib.load(MODELS_DIR / "ensemble_full.pkl"),
    ("ensemble_v2", "80"):   joblib.load(MODELS_DIR / "ensemble_v2_80.pkl"),
    ("ensemble_v2", "full"): joblib.load(MODELS_DIR / "ensemble_v2_full.pkl"),
}

# Case-insensitive brand lookup (Issue 2 fix — see Step 22)
brand_lookup = {b.lower(): b for b in encoders["brand_name"].classes_}

# Pre-compute encoded value for "unknown" brand once
UNKNOWN_BRAND_ENC = int(encoders["brand_name"].transform(["unknown"])[0])

print(f"Loaded 10 models + 6 preprocessing objects. "
      f"Brand lookup: {len(brand_lookup):,} entries.")


# =====================================================================
# SECTION 2 — PIPELINE
# =====================================================================

def clean_text(text):
    """Lowercase + normalize whitespace. Same as Phase I Cell 23."""
    if not isinstance(text, str) or text.strip() == "":
        return "no description"
    text = text.lower()
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_encode(value, encoder, fallback):
    """Encode value; fall back to fallback if value was never seen in training."""
    target = value if value in encoder.classes_ else fallback
    return int(encoder.transform([target])[0])


def listing_to_features(name, item_condition_id, category_name,
                         brand_name, shipping, item_description):
    """
    Raw listing -> (X_full_row [1 x 100,012 sparse], X_lgbm_row [1 x 112 dense])

    X_full_row  : Ridge | LightGBM_full (lgbm_text) | Ensemble v2
    X_lgbm_row  : LightGBM+SVD (lgbm) | Ensemble v1
    """
    name_clean = clean_text(name)
    desc_clean = clean_text(item_description)

    parts = [p.strip().lower() for p in (category_name or "").split("/", 2)]
    while len(parts) < 3:
        parts.append("missing")
    cat_1, cat_2, cat_3 = parts[:3]

    brand_raw = (brand_name or "").strip()
    if brand_raw == "":
        brand = "unknown"
    elif brand_raw in encoders["brand_name"].classes_:
        brand = brand_raw
    else:
        brand = brand_lookup.get(brand_raw.lower(), brand_raw)

    cat_1_enc = safe_encode(cat_1, encoders["cat_1"], "missing")
    cat_2_enc = safe_encode(cat_2, encoders["cat_2"], "missing")
    cat_3_enc = safe_encode(cat_3, encoders["cat_3"], "missing")
    brand_enc = safe_encode(brand, encoders["brand_name"], "unknown")

    numeric_vec = np.array([[
        item_condition_id, shipping,
        cat_1_enc, cat_2_enc, cat_3_enc, brand_enc,
        len(name_clean), len(desc_clean),
        len(name_clean.split()), len(desc_clean.split()),
        0 if desc_clean == "no description" else 1,
        0 if brand_enc == UNKNOWN_BRAND_ENC else 1,
    ]], dtype=np.float64)

    numeric_scaled = scaler.transform(numeric_vec)

    name_vec = tfidf_name.transform([name_clean])
    desc_vec  = tfidf_desc.transform([desc_clean])

    # X_full: 12 scaled numeric + 50K name TF-IDF + 50K desc TF-IDF
    X_full_row = sp.hstack([
        sp.csr_matrix(numeric_scaled), name_vec, desc_vec
    ]).tocsr()

    # X_lgbm_row: 12 scaled numeric + 50 name-SVD topics + 50 desc-SVD topics
    name_svd   = svd_name.transform(name_vec)
    desc_svd   = svd_desc.transform(desc_vec)
    X_lgbm_row = np.hstack([numeric_scaled, name_svd, desc_svd])

    return X_full_row, X_lgbm_row


def predict_price(listing: dict, model_type: str, data_version: str) -> float:
    """
    model_type   : 'ridge' | 'lgbm' | 'lgbm_text' | 'ensemble' | 'ensemble_v2'
    data_version : '80' | 'full'
    """
    X_full_row, X_lgbm_row = listing_to_features(**listing)

    if model_type == "ridge":
        # Ridge regression on 100K TF-IDF + scaled numeric (linear)
        log_pred = MODELS[("ridge", data_version)].predict(X_full_row)[0]

    elif model_type == "lgbm":
        # LightGBM on 112 SVD-compressed text + numeric features
        log_pred = MODELS[("lgbm", data_version)].predict(X_lgbm_row)[0]

    elif model_type == "lgbm_text":
        # LightGBM on full 100K TF-IDF + numeric (feature_fraction=0.3)
        log_pred = MODELS[("lgbm_text", data_version)].predict(X_full_row)[0]

    elif model_type == "ensemble":
        # v1: 0.7 x Ridge + 0.3 x LightGBM+SVD
        bundle = MODELS[("ensemble", data_version)]
        r = bundle["ridge_model"].predict(X_full_row)[0]
        l = bundle["lgbm_model"].predict(X_lgbm_row)[0]
        log_pred = bundle["w_ridge"] * r + bundle["w_lgbm"] * l

    elif model_type == "ensemble_v2":
        # v2: 0.4 x Ridge + 0.6 x LightGBM_full — best config (RMSLE 0.443)
        bundle = MODELS[("ensemble_v2", data_version)]
        r = bundle["ridge_model"].predict(X_full_row)[0]
        l = bundle["lgbm_full_model"].predict(X_full_row)[0]
        log_pred = bundle["w_ridge"] * r + bundle["w_lgbm_full"] * l

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return float(np.expm1(log_pred))


# =====================================================================
# SECTION 3 — REQUEST SCHEMA
# =====================================================================

class ListingRequest(BaseModel):
    name: str = Field(..., min_length=1,
                      examples=["Nike Air Force 1 white sneakers size 9"])
    item_condition_id: int = Field(..., ge=1, le=5, examples=[2])
    category_name: str  = Field(default="", examples=["Men/Shoes/Athletic"])
    brand_name: str     = Field(default="", examples=["Nike"])
    shipping: int       = Field(..., ge=0, le=1, examples=[1])
    item_description: str = Field(default="",
                      examples=["Worn a few times, great condition"])
    model_type: str    = Field(
        default="ensemble_v2",
        pattern="^(ridge|lgbm|lgbm_text|ensemble|ensemble_v2)$"
    )
    data_version: str  = Field(
        default="full",
        pattern="^(80|full)$"
    )


class PredictionResponse(BaseModel):
    predicted_price: float
    model_type: str
    data_version: str


# =====================================================================
# SECTION 4 — ROUTES
# =====================================================================

app = FastAPI(title="Mercari Price Suggestion API")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_model=PredictionResponse)
def predict(req: ListingRequest):
    listing = {
        "name":              req.name,
        "item_condition_id": req.item_condition_id,
        "category_name":     req.category_name,
        "brand_name":        req.brand_name,
        "shipping":          req.shipping,
        "item_description":  req.item_description,
    }
    price = predict_price(listing, req.model_type, req.data_version)
    return PredictionResponse(
        predicted_price=round(price, 2),
        model_type=req.model_type,
        data_version=req.data_version,
    )