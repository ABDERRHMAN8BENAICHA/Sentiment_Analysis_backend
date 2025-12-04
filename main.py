from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_model.joblib"
TFIDF_PATH = BASE_DIR / "tfidf.joblib"

model = None
tfidf = None

app = FastAPI(title="Sentiment Analysis API")
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def load_artifacts():
    global model, tfidf
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found. Place `svm_model.joblib` and `tfidf.joblib` in {BASE_DIR}"
        ) from e

def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    if model is None or tfidf is None:
        raise RuntimeError("Model not loaded")
    text = clean(review.text)
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    label = "positive" if pred == 1 else "negative"
    return {"sentiment": label}