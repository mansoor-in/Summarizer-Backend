from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import random

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def augment_input(text):
    sentences = text.split(". ")
    random.shuffle(sentences)
    augmented_text = ". ".join(sentences).strip()
    return augmented_text

def abstractive_summarizer(text):
    try:
        if not text.strip():
            return "Error: Input text is empty. Please provide valid text."
        
        cleaned_text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text).strip()
        
        if len(cleaned_text.split()) < 3:
            return "Error: Input text contains insufficient meaningful content. Please provide valid text."
        
        augmented_text = augment_input(cleaned_text)
        input_text = "summarize: " + augmented_text
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        summary_ids = model.generate(
            inputs,
            max_length=100,
            min_length=30,
            temperature=1.7,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            num_return_sequences=1,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        return f"An error occurred: {str(e)}"

# API route to summarize text
@app.route("/", methods=["POST"])
def summarize():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    summary = abstractive_summarizer(text)
    return jsonify({"summary": summary}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
