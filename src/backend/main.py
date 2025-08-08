from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.backend.model import SentimentAnalyzer, LLMSentimentAnalyzer

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str
    model: str

class SentimentResponse(BaseModel):
    sentiment: str

# Cache models
model_cache = {
    "default": None,
    "llm": None
}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    text = request.text
    model_type = request.model.lower()

    if model_type not in model_cache:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'default' or 'llm'.")

    # Load model lazily and cache
    if model_cache[model_type] is None:
        if model_type == "default":
            model_cache["default"] = SentimentAnalyzer()
        elif model_type == "llm":
            model_cache["llm"] = LLMSentimentAnalyzer()

    model = model_cache[model_type]
    sentiment = model.predict(text)
    return SentimentResponse(sentiment=sentiment)
