## 📞 Call Sentiment Analyzer

This project classifies customer support call transcripts into sentiment labels: **Positive**, **Neutral**, or **Negative**.

It provides:

* ✅ Two models to choose from:

  * `default`: A simple DNN classifier
  * `llm`: A small LLM-based classifier (e.g., Gemma 2B)
* 🌐 REST API using FastAPI
* 🖥️ Web UI using Streamlit
* 🐳 Docker support

---

## 🔧 How to Use

### 🐳 Option 1: Run with Docker

1. Build the image:

```bash
docker build -t call-sentiment-analyzer .
```

2. Run backend API:

```bash
docker run -p 8000:8000 call-sentiment-analyzer
```

3. (Optional) Run frontend UI:

```bash
docker run -p 8501:8501 call-sentiment-analyzer streamlit run src/frontend/app.py
```

---

### 💻 Option 2: Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run backend:

```bash
uvicorn src.backend.main:app --reload
```

3. Run frontend:

```bash
streamlit run src/frontend/app.py
```

---

## 📤 API Example

POST `/predict`
Example request:

```json
{
  "text": "العميل غير راضٍ عن الخدمة",
  "model": "llm"
}
```

Response:

```json
{
  "sentiment": "Negative"
}
```

---

## 🐳 Docker Notes

* This repo includes a `Dockerfile` for easy containerization.
* Both the backend and frontend can be run in containers.
* You can push the image to Docker Hub if needed.


## Contact
For any inquiries or support, please contact yahyadaqour@gmail.com.