from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
from datetime import datetime
import sqlite3


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
init_db()


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Save upload record in database
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO uploads (filename, timestamp) VALUES (?, ?)",
                (file.filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()

        # Placeholder prediction (no TensorFlow locally)
        # Later, when deploying on Render, you’ll load your .h5 model there.
        emotion = "Prediction pending (model not loaded locally)"
        return render_template("index.html", emotion=emotion, image=file.filename)
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
