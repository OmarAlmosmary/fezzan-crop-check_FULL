"""Predict the crop group from one or more ground photos, and check a reported crop.

Usage:
    python predict_photo.py path/to/photo.jpg [more photos ...]
    python predict_photo.py path/to/photo.jpg --reported cereals      # flag if the reported crop is very unlikely

Needs outputs/models_photo/ (written by fezzan_photos_part3.ipynb).
"""
import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, keras, joblib
from PIL import Image

MODELS = "outputs/models_photo"
WEIGHTS_DIR = "weights"
BACKBONES = {
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"),
    "densenet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input, "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"),
}


def load(path, px=224):
    """Shorter side to px, then centre square crop (same as the notebook)."""
    im = Image.open(path).convert("RGB"); w, h = im.size; s = px / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR); w, h = im.size; l, t = (w - px) // 2, (h - px) // 2
    return np.asarray(im.crop((l, t, l + px, t + px)), dtype=np.uint8)


class PhotoModel:
    def __init__(self, models=MODELS):
        cfg = json.load(open(f"{models}/config.json")); self.classes, self.tl = cfg["classes"], cfg["transfer"]
        fn, self.pre, wfile = BACKBONES[self.tl["backbone"]]; local = os.path.join(WEIGHTS_DIR, wfile)
        self.base = fn(include_top=False, weights=None if os.path.exists(local) else "imagenet", input_shape=(224, 224, 3), pooling="avg")
        if os.path.exists(local): self.base.load_weights(local)
        if self.tl["head"] == "logistic": self.head = joblib.load(f"{models}/transfer_head.joblib"); self.scale = None
        else:
            self.head = keras.models.load_model(f"{models}/transfer_head.keras"); s = np.load(f"{models}/transfer_scaler.npz"); self.scale = (s["mu"], s["sd"])

    def predict(self, paths):
        X = np.stack([load(p) for p in paths]); views = (X, X[:, :, ::-1]) if self.tl["flips"] else (X,)
        P = []
        for V in views:
            f = self.base.predict(self.pre(V.astype("float32")), verbose=0)
            P.append(self.head.predict_proba(f) if self.scale is None else self.head.predict((f - self.scale[0]) / self.scale[1], verbose=0))
        return {c: round(float(v), 3) for c, v in zip(self.classes, np.mean(P, axis=0).mean(0))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("photos", nargs="+"); ap.add_argument("--reported", choices=["alfalfa", "cereals", "palm", "palm_orchard"])
    a = ap.parse_args(); p = PhotoModel().predict(a.photos); print(json.dumps(p, indent=1))
    if a.reported:
        best = max(p, key=p.get); flag = p[a.reported] < 0.15 and p[best] >= 0.6
        print(f"reported: {a.reported} | model: {best} ({p[best]:.2f}) | {'FLAG for review' if flag else 'ok'}")
