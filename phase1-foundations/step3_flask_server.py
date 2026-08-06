# step3_flask_server.py
#
# GOAL: Wrap our sentiment model in a Flask server, so other programs
# (browser, curl, React frontend later) can send it text and get back
# a prediction -- over a network, using an API.
#
# Run this with:   python3 step3_flask_server.py
# Then, in ANOTHER terminal window, test it with:
#   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"I love this!\"}"

from flask import Flask, request, jsonify
from transformers import pipeline

# --- Step 1: Create the Flask application ---
# This "app" object is what represents our server. We attach behavior
# to it below.
app = Flask(__name__)

# --- Step 2: Load the model ONCE, when the server starts ---
# We do NOT want to reload the model every time someone sends a request --
# that would be extremely slow. So we load it once, up front, and keep
# it in memory (in the `classifier` variable) for as long as the server runs.
print("Loading model, please wait...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded! Server starting...")


# --- Step 3: Define what happens when someone sends a POST to /predict ---
# The "@app.route(...)" line above the function is called a "decorator" --
# it's Python's way of saying "connect this URL path and HTTP verb to
# the function right below it."
@app.route("/predict", methods=["POST"])
def predict():
    # `request.json` reads the JSON data that was sent to us.
    # We expect something like: {"text": "this movie was amazing"}
    data = request.json
    text = data.get("text", "")

    if not text:
        # If no text was sent, respond with an error.
        # 400 is the standard HTTP status code for "bad request" --
        # i.e. the client sent us something we can't use.
        return jsonify({"error": "No 'text' field provided"}), 400

    # Run the actual model inference (same as step 2!)
    result = classifier(text)[0]

    # Send back a JSON response.
    # jsonify() converts a Python dictionary into a proper JSON HTTP response.
    return jsonify({
        "text": text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    })


# --- Step 4: Actually start the server ---
if __name__ == "__main__":
    # debug=True gives helpful error messages while we're developing.
    # port=5000 means our server listens at http://127.0.0.1:5000
    app.run(debug=True, port=5000)
