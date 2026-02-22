from flask import Flask, render_template, request
import json
import random
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------- DATABASE SETUP ----------
def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- LOAD INTENTS ----------
with open('intents.json') as file:
    data = json.load(file)

patterns = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(random.choice(intent['responses']))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"]
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()

    if similarity[0][index] > 0.3:
        bot_reply = responses[index]
    else:
        bot_reply = "Sorry, I could not understand. Please describe your symptom clearly."

    # SAVE TO DATABASE
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (user_message, bot_response) VALUES (?, ?)",
                   (user_input, bot_reply))
    conn.commit()
    conn.close()

    return bot_reply

if __name__ == "__main__":
    app.run(debug=True)

