from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from datetime import datetime
import sqlite3
from werkzeug.utils import secure_filename


def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    )
    conn.commit()
    conn.close()


# Initialize database automatically
# Initialize database automatically
init_db()


app = Flask(__name__)

# Store uploads inside the static folder so Flask can serve them easily
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")
else:
    print(f"Upload folder exists: {UPLOAD_FOLDER}")

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database if not exists
db = "database.db"
if not os.path.exists(db):
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS uploads
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         timestamp TEXT)"""
        )

# Try to load a TensorFlow model if available. Import TF inside try so the
# app can still start locally when TF isn't installed.
MODEL_PATH = os.path.join(app.root_path, "face_emotionModel.h5")
model = None
tf = None
try:
    import tensorflow as _tf

    tf = _tf
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Loaded model:", MODEL_PATH)
        except Exception as e:
            print("Model found but failed to load:", e)
    else:
        print("Model file not found at", MODEL_PATH)
except Exception as e:
    # TensorFlow not installed or failed to import. We'll keep model=None.
    print("TensorFlow not available or failed to import:", e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("image")
        if not file:
            print("No file received")
            return redirect(url_for("home"))

        if not file.filename:
            print("File has no filename")
            return render_template("index.html", emotion="No file selected")

        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return render_template("index.html", emotion="Invalid file type")

        filename = secure_filename(file.filename)
        if not filename:
            print("Filename became empty after sanitization")
            return render_template("index.html", emotion="Invalid filename")
            
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"Saved file to: {filepath}")

        # Save upload record in database
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO uploads (filename, timestamp) VALUES (?, ?)",
                (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()

        # If the model is loaded, run a simple preprocessing + prediction path.
        emotion = "Prediction pending (model not loaded)"
        if model is not None and tf is not None:
            try:
                # Example preprocessing for fer2013-like models: grayscale 48x48
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    emotion = "Failed to read uploaded image"
                else:
                    img = cv2.resize(img, (48, 48))
                    img = img.astype("float32") / 255.0

                    # If model expects channels last with 1 channel
                    if len(model.input_shape) == 4 and model.input_shape[-1] == 1:
                        img = np.expand_dims(img, -1)

                    img = np.expand_dims(img, 0)
                    preds = model.predict(img)
                    idx = int(np.argmax(preds))
                    mapping = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
                    if 0 <= idx < len(mapping):
                        emotion = mapping[idx]
                    else:
                        emotion = f"Unknown prediction index {idx}"
            except Exception as e:
                print(f"Prediction error: {e}")
                emotion = f"Prediction error: {e}"

        return render_template("index.html", emotion=emotion, image=filename)
    
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", emotion=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
