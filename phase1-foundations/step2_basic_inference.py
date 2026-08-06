# step2_basic_inference.py
#
# GOAL: Load a real, pretrained machine learning model and use it to
# classify text as POSITIVE or NEGATIVE sentiment. No API, no distribution
# yet -- just proving we can talk to a real model.
#
# Run this with:   python3 step2_basic_inference.py

# "transformers" is a library from HuggingFace that gives us easy access
# to pretrained models (models someone else already trained on huge datasets).
# "pipeline" is a helper that bundles together: loading the model,
# preparing your text for it, running it, and turning the output into
# something human-readable.
from transformers import pipeline

def load_model():
    """
    Loads a small, pretrained sentiment-analysis model.
    'distilbert-base-uncased-finetuned-sst-2-english' is a real model
    trained to classify English text as POSITIVE or NEGATIVE.

    The first time you run this, it will DOWNLOAD the model (a few
    hundred MB) from huggingface.co. After that, it's cached locally,
    so future runs are fast.
    """
    print("Loading model... (first run will download it, be patient)")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded!")
    return classifier


def run_inference(classifier, text):
    """
    Takes a loaded model (classifier) and a piece of text,
    returns the model's prediction.
    """
    result = classifier(text)
    # result looks like: [{'label': 'POSITIVE', 'score': 0.999}]
    return result[0]


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # (a Python convention -- similar to a main() function/entry point
    # in other languages).

    classifier = load_model()

    # A few example sentences to test with
    test_sentences = [
        "This project is going great!",
        "I am so frustrated with this bug.",
        "The weather today is fine, nothing special.",
    ]

    for sentence in test_sentences:
        prediction = run_inference(classifier, sentence)
        print(f"Text: {sentence!r}")
        print(f"  -> Sentiment: {prediction['label']}, Confidence: {prediction['score']:.2f}")
        print()
