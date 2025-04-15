# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Standard library and external package imports
import os
import re
import string
import logging
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import psycopg2
from psycopg2.extras import execute_values
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import ollama

# Download NLTK stopwords (Turkish)
nltk.download('stopwords')
turkish_stopwords = set(stopwords.words('turkish'))

# Load configuration from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATA_DIR = os.getenv("DATA_DIR")
ALLOWED_IP = os.getenv("ALLOWED_IP", "127.0.0.1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CORS_ORIGINS = os.getenv("CORS_ORIGINS")

# Load embedding model and zero-shot classifier
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2"))
classifier = pipeline("zero-shot-classification", model=os.getenv("CLASSIFIER_MODEL", "facebook/bart-large-mnli"))

# Configure Flask app and CORS policy
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app, resources={r"/chat/*": {"origins": CORS_ORIGINS}})

# Restrict access to a single allowed IP address
@app.before_request
def limit_remote_addr():
    if request.remote_addr != ALLOWED_IP:
        abort(403)

# Identify categories from subfolders in the data directory
categories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# Extract top keywords for each category (from .txt files)
def extract_keywords_for_category(category_name, path, top_k=10):
    words = []
    for file in os.listdir(path):
        if file.endswith(".txt"):
            full_path = os.path.join(path, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    content = content.translate(str.maketrans('', '', string.punctuation))
                    tokens = content.split()
                    words += [tok for tok in tokens if tok not in turkish_stopwords and len(tok) > 2]
            except Exception as e:
                print(f"[{category_name}] '{file}' could not be read: {e}")
    most_common = Counter(words).most_common(top_k)
    return [word for word, _ in most_common]

# Category example labels for zero-shot classification
category_examples = {
    category: f"Content related to {category}" for category in categories
}

# Top keywords per category
category_keywords = {
    category: extract_keywords_for_category(
        category, os.path.join(DATA_DIR, category), top_k=10
    )
    for category in categories
}

# Embedding representation of category labels
category_embeddings = {
    category: embedding_model.encode(description, normalize_embeddings=True)
    for category, description in category_examples.items()
}

# Cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b)

# Keyword match scoring
def keyword_match_score(query, keywords):
    query = query.lower()
    score = sum(1 for k in keywords if k in query)
    return score / len(keywords)

# Hybrid category selection: combines cosine, keyword, and zero-shot classification
def select_category_hybrid_zero(query):
    query_vec = embedding_model.encode(query, normalize_embeddings=True)
    zero_shot_results = classifier(query, list(category_examples.keys()), multi_label=True)
    zero_shot_score_map = {
        label: score for label, score in zip(zero_shot_results["labels"], zero_shot_results["scores"])
    }

    scores = {}
    for category, cat_vec in category_embeddings.items():
        cosine_score = cosine_similarity(query_vec, cat_vec)
        keyword_score = keyword_match_score(query, category_keywords.get(category, []))
        zero_score = zero_shot_score_map.get(category, 0.0)
        final_score = 0.4 * cosine_score + 0.3 * keyword_score + 0.3 * zero_score
        scores[category] = final_score

    logging.debug(f"[SCORES] {scores}")
    max_score = max(scores.values())
    selected = [k for k, v in scores.items() if v >= max_score - 0.05]
    logging.info(f"[Hybrid-ZeroShot] Selected categories: {selected}")
    return selected

# Find the closest matching categories based on cosine similarity
def get_closest_categories(query_vec, threshold=0.85):
    selected = []
    for category, cat_vec in category_embeddings.items():
        similarity = cosine_similarity(query_vec, cat_vec)
        if similarity >= threshold:
            selected.append((category, similarity))

    if not selected:
        best = max(category_embeddings.items(), key=lambda x: cosine_similarity(query_vec, x[1]))
        selected = [(best[0], cosine_similarity(query_vec, best[1]))]

    selected.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in selected]

# Extract clean message text from Ollama API response
def extract_message_content(response):
    try:
        if isinstance(response, dict) and 'message' in response:
            content = response['message'].get('content', '')
        elif hasattr(response, 'message'):
            content = getattr(response.message, 'content', '')
        else:
            logging.warning(f"Unexpected response structure: {response}")
            return "Error occurred while processing the response"

        content = re.sub(r"<think>[\s\S]*?</think>\s*\n*", "", content)
        return content.strip()

    except Exception as e:
        logging.exception("Response processing error:")
        return "An error occurred while processing the response"

# Retrieve relevant context from PostgreSQL using embedding vector search
def retrieve_context_from_vector(query):
    try:
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        selected_categories = select_category_hybrid_zero(query)

        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()

        contents = []
        for category in selected_categories:
            cur.execute(
                "SELECT content FROM data WHERE category = %s ORDER BY embedding <#> %s::vector LIMIT 5",
                (category, query_embedding.tolist())
            )
            contents += [row[0] for row in cur.fetchall()]

        cur.close()
        conn.close()
        return contents

    except Exception as e:
        logging.exception("[retrieve_context_from_vector] Error:")
        return []

# Home route (health check)
@app.route("/")
def home():
    return "Flask API is running. Use the /chat endpoint for POST requests.", 200

# Route for chat UI (GET)
@app.route("/chat", methods=["GET"])
def chat_ui():
    return render_template("chat.html")

# Main chat POST endpoint: processes message history and returns AI response
@app.route("/chat", methods=["POST"])
def chat_api():
    messages = request.json.get('messages')

    if not messages or not isinstance(messages, list):
        return jsonify({'error': 'Messages must be a list'}), 400

    try:
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        logging.debug(f"User message received: {user_message}")

        if not user_message.strip():
            return jsonify({'error': 'Last user message cannot be empty'}), 400

        context = retrieve_context_from_vector(user_message)
        context_text = "\n".join(context) if context else ""

        if not context or len(context_text.strip()) < 50:
            return jsonify({'response': "No information available on this topic."})

        messages = [
            {
                "role": "system",
                "content": f"""
        You are an information assistant and can only answer questions using the context below. 
        Do not make guesses, comments, or fabricate any information. 
        Your answers must be objective, concise, and informative.

        If the user's question is not covered in the context, respond with: "No information available on this topic."

        The following context is taken from NGN company documents:

        {context_text}
                """
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1}
        )

        response_text = extract_message_content(response)
        logging.debug(f"Ollama response text:\n{response_text}")
        return jsonify({'response': response_text})

    except Exception as e:
        logging.exception("An error occurred:")
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
