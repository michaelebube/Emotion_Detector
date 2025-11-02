# FACE_DETECTION/prepare_fer2013.py
import numpy as np
import pandas as pd
import os

CSV_PATH = "fer2013.csv"  # put fer2013.csv in the project root or update this path
OUT_NPZ = "fer2013_data.npz"


def load_fer2013(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    # Columns typically: emotion, pixels, Usage
    pixels = df["pixels"].tolist()
    emotions = df["emotion"].astype(int).values
    usages = df["Usage"].values if "Usage" in df.columns else None

    X = np.zeros((len(pixels), 48, 48), dtype=np.float32)
    for i, px in enumerate(pixels):
        arr = np.fromstring(px, dtype=np.uint8, sep=" ")
        if arr.size != 2304:
            raise ValueError(f"Row {i} has unexpected size {arr.size}")
        X[i] = arr.reshape(48, 48)

    # Normalize to 0-1 and add channel dimension for Keras (H, W, C)
    X = X / 255.0
    X = X[..., np.newaxis]  # shape: (N, 48, 48, 1)

    if usages is not None:
        train_idx = np.where(usages == "Training")[0]
        val_idx = np.where(usages == "PublicTest")[0]
        test_idx = np.where(usages == "PrivateTest")[0]

        X_train, y_train = X[train_idx], emotions[train_idx]
        X_val, y_val = X[val_idx], emotions[val_idx]
        X_test, y_test = X[test_idx], emotions[test_idx]
    else:
        # fallback random split
        n = len(X)
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

        X_train, y_train = X[train_idx], emotions[train_idx]
        X_val, y_val = X[val_idx], emotions[val_idx]
        X_test, y_test = X[test_idx], emotions[test_idx]

    np.savez_compressed(
        OUT_NPZ,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print("Saved:", OUT_NPZ)
    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Put fer2013.csv next to this script."
        )
    load_fer2013()
