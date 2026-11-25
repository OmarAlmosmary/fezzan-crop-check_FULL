"""Predict the crop group of one farm from one or more 512 x 512 satellite views.

Usage:
    python predict.py path/to/view1.jpg [path/to/view2.jpg ...]
    python predict.py --check data/chips_index.csv --out review_list.csv   # survey check for every farm in an index file

Needs the files written by the notebook in outputs/models/ (config.json and the transfer head).
"""
import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, keras
from PIL import Image

MODELS = "outputs/models"
WEIGHTS_DIR = "weights"
BACKBONES = {
    "resnet50": (keras.applications.ResNet50, keras.applications.resnet50.preprocess_input, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"),
    "densenet121": (keras.applications.DenseNet121, keras.applications.densenet.preprocess_input, "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"),
}


class CropModel:
    def __init__(self, models=MODELS):
        cfg = json.load(open(f"{models}/config.json"))
        self.classes, self.tl = cfg["classes"], cfg["transfer"]
        fn, self.pre, wfile = BACKBONES[self.tl["backbone"]]
        local = os.path.join(WEIGHTS_DIR, wfile)
        self.base = fn(include_top=False, weights=None if os.path.exists(local) else "imagenet", input_shape=(224, 224, 3), pooling="avg")
        if os.path.exists(local): self.base.load_weights(local)
        if self.tl["head"] == "logistic":
            import joblib
            self.head = joblib.load(f"{models}/transfer_head.joblib"); self.scale = None
        else:
            self.head = keras.models.load_model(f"{models}/transfer_head.keras")
            s = np.load(f"{models}/transfer_scaler.npz"); self.scale = (s["mu"], s["sd"])

    def _load(self, path):
        im = Image.open(path).convert("RGB")
        if im.size != (512, 512): im = im.resize((512, 512))
        if self.tl["ctx"] == 290: im = im.crop((128, 128, 384, 384))
        return np.asarray(im.resize((224, 224), Image.BILINEAR), dtype=np.uint8)

    def predict_farm(self, paths):
        X = np.stack([self._load(p) for p in paths])
        rots = range(4) if self.tl["rotations"] else range(1)
        probs = []
        for r in rots:
            f = self.base.predict(self.pre(np.rot90(X, r, axes=(1, 2)).astype("float32")), verbose=0)
            probs.append(self.head.predict_proba(f) if self.scale is None else self.head.predict((f - self.scale[0]) / self.scale[1], verbose=0))
        return {c: round(float(v), 3) for c, v in zip(self.classes, np.mean(probs, axis=0).mean(0))}


def survey_check(model, index_csv, data_dir="data", p_reported_max=0.15, p_predicted_min=0.6):
    idx = pd.read_csv(index_csv); rows = []
    for farm, g in idx.groupby("farm"):
        p = model.predict_farm([os.path.join(data_dir, x) for x in g.image])
        pred = max(p, key=p.get); rep = g.label.iloc[0]
        rows.append({"farm": farm, "reported": rep, "predicted": pred, "p_reported": p.get(rep, float("nan")), "p_predicted": p[pred],
                     "flag": p.get(rep, 1) < p_reported_max and p[pred] >= p_predicted_min})
    return pd.DataFrame(rows).sort_values(["flag", "p_reported"], ascending=[False, True])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*"); ap.add_argument("--check"); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="review_list.csv")
    a = ap.parse_args(); m = CropModel()
    if a.check:
        r = survey_check(m, a.check, a.data); r.to_csv(a.out, index=False)
        print(f"{len(r)} farms checked, {int(r.flag.sum())} flagged -> {a.out}")
    elif a.images:
        print(json.dumps(m.predict_farm(a.images), indent=1))
    else:
        ap.print_help()
