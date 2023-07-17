from flask import Flask, request, jsonify, render_template
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the pre-trained model
model = SentenceTransformer("stsb-roberta-large")

# Load the data into a DataFrame


# Home page with form to input text1 and text2
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# Endpoint for similarity calculation
@app.route("/calculate-similarity", methods=["POST"])
def calculate_similarity():
    # Get the input data from the form
    text1 = request.form["text1"]
    text2 = request.form["text2"]

    # Encode sentences to get their embeddings
    sentence_embeddings = model.encode([text1, text2])

    # Calculate the similarity score
    similarity_score = 1 - cosine(sentence_embeddings[0], sentence_embeddings[1])

    # Return the response as JSON
    response = {"similarity score": similarity_score}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=False)
