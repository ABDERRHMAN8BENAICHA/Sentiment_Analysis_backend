from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional
import warnings
from contextlib import asynccontextmanager

# Import concrete sklearn types for better static analysis (optional)
try:
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    # if sklearn isn't importable at edit-time, fall back to Any types
    LinearSVC = Any  # type: ignore
    TfidfVectorizer = Any  # type: ignore

# Optionally suppress scikit-learn version mismatch warnings (use with caution)
try:
    from sklearn.base import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    # sklearn may not be installed yet in the environment where this file is being edited
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.joblib"
TFIDF_PATH = BASE_DIR / "tfidf.joblib"

model: Optional[LinearSVC] = None
tfidf: Optional[TfidfVectorizer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler to load model artifacts on startup.

    Using lifespan is the recommended approach over the deprecated `@app.on_event("startup")`.
    """
    global model, tfidf
    try:
        if not MODEL_PATH.exists() or not TFIDF_PATH.exists():
            raise FileNotFoundError("One or both model files are missing")
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found. Place `svm_model.joblib` and `tfidf.joblib` in {BASE_DIR}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}") from e

    yield
    # no cleanup required


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# configure CORS - adjust origins to match your frontend
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://sentiment-analysis-three-chi.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean(text: str) -> str:
    """Basic text cleaning used before vectorization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Review(BaseModel):
    text: str


@app.get("/", tags=["health"])
def root() -> dict:
    return {"status": "ok", "model_loaded": model is not None and tfidf is not None}


@app.post("/predict")
def predict_sentiment(review: Review) -> dict:
    if model is None or tfidf is None:
        # Prefer an HTTPException so the client gets a 5xx response instead of the process crashing.
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    text = clean(review.text)
    try:
        X = tfidf.transform([text])
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    label = "positive" if int(pred) == 1 else "negative"
    return {"sentiment": label}