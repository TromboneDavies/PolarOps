
import joblib    # to save/restore models
from pathlib import Path
import sys

if not Path("skpolarops.joblib").is_file():
    sys.exit("No skpolarops.joblib model file.")
else:
    classifier = joblib.load("skpolarops.joblib")
    sent = input("enter a sentence (or 'done'): ")
    while sent != 'done':
        pred = "polar" if classifier.predict([sent])[0] else "nonpolar"
        print(f"Prediction: {pred}")

        sent = input("enter a sentence (or 'done'): ")
