import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


class SentimentAnalyzer:
    def __init__(self, model_name="tabularisai/multilingual-sentiment-analysis"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["Negative", "Neutral", "Positive"]

    def predict(self, text: str) -> str:
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_map = {0: "Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Positive"}
        return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()][0]
    
class LLMSentimentAnalyzer:
  def __init__(self, model_name="google/gemma-2b-it"):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
    self.labels = ["Negative", "Neutral", "Positive"]

  def get_prompt(self, text: str) -> str:
      prompt = f"""
      You are a sentiment analysis assistant for customer service calls. Your task is to classify the sentiment of the given call summary in Arabic or English.

      Please consider the tone, language, and content of the summary when making your classification. You must choose one of the following:
      - Positive
      - Neutral
      - Negative

      Here are some examples to make your classification:

      Example 1:
      Text: The customer was experiencing an issue with the laptop that has not been resolved yet after several attempts, and an appointment was scheduled to follow up on the case next Sunday.
      Sentiment: Negative

      Example 2:
      Text: تواصل العميل مع مركز خدمة العمالء للحصول على شهادة رصيد حسابها. تم توجيهها إلى الموقع اإللكتروني للبنك إلصدار الشهادة وتوجيهها إلى جهة معينة
      Sentiment: Neutral

      Example 3:
      Text: كان العميل يواجه صعوبة في سماع الوكيل أثناء المكالمة بسبب انخفاض مستوى الصوت. وافق الوكيل على إرسال رسالة نصية عبر تطبيق واتساب وأبدى العميل امتنانه بينما ينتظر اتصاالً معاوداً
      Sentiment: Positive
      Your classification should be based solely on the content of the call summary. Do not make any assumptions or inferences beyond what is explicitly stated.

      Classify the sentiment of the following call summary:

      {text}

      Answer: """

      return prompt

  def extract_sentiment(self, output: str) -> str:
    lines = output.splitlines()
    last_line = lines[-1]
    for label in self.labels:
      if label in last_line:
        return label
    return "Unknown"

  def predict(self, text: str) -> str:
    prompt = self.get_prompt(text)
    input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model.generate(**input_ids)
    decoded_output = self.tokenizer.decode(outputs[0])
    
    sentiment_answer = self.extract_sentiment(decoded_output)
    
    return sentiment_answer