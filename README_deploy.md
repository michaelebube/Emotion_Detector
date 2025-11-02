Deploy notes for Render

1. Requirements

- This project already contains `requirements.txt`. It includes `tensorflow==2.15.0` which will be installed on Render and can increase build time and memory usage.
- Keep `face_emotionModel.h5` in the project root (already present) so the app can load it at startup.

2. Files added

- `Procfile` - tells Render to run `gunicorn app:app`.
- `runtime.txt` - pins Python version to 3.10.12.

3. Important app behavior

- Uploaded images are saved to `static/uploads/` and served via `url_for('static', filename='uploads/<name>')`.
- The app attempts to import TensorFlow at startup. If TF is unavailable or the model fails to load, the app will still start and return a friendly fallback message on uploads.

4. Deploy steps (high level)

1) Initialize a git repository and commit your code.
   git init; git add .; git commit -m "initial"
2) Push the repo to GitHub (or connect your existing repo to Render).
3) On Render.com create a new Web Service, connect the repository, and set the build command to the default (Render will run pip install -r requirements.txt).
4) For the start command, Render detects the `Procfile` so no change is necessary. If required set: `gunicorn app:app`

Notes:

- TensorFlow is large; the first deployment may take longer and require more memory. If you encounter OOM during build or runtime, consider using a smaller TF build (e.g., `tensorflow-cpu`) or using Render's larger instance types.
- If you need GPU inference on Render, you must select a GPU instance type and ensure compatible TF/CUDA versions. That is an advanced step and outside these quick notes.
