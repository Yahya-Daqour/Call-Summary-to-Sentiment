import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("📞 Call Sentiment Analyzer")

# Input text box
text_input = st.text_area("Enter call summary:", height=200)

# Model selection
model_choice = st.radio("Choose Model:", ("Default", "LLM"))

# Submit button
if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": text_input, "model": model_choice.lower()}
            )
            if response.status_code == 200:
                sentiment = response.json().get("sentiment")
                st.success(f"Predicted Sentiment: **{sentiment}**")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
