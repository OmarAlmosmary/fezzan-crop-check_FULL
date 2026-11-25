"""Batch 3 view selection: same rules as tools/select_views.py; cursor offsets of this batch measured from the OCR."""
import numpy as np, pandas as pd
d = pd.read_pickle("ocr3_all.pkl")
z = np.load("thumbs3.npz"); T = z["t"].astype(float); F = [f + ".png" for f in z["names"]]
d = d.set_index("file").loc[F].reset_index(); d["i"] = range(len(d))
CURSOR = np.array([(-240, 55), (-65, -100)])
def status(r):
    if pd.isna(r.dist): return "unread"
    if r.dist <= 60: return "centre"
    return "cursor" if np.min(np.hypot(CURSOR[:, 0] - r.east, CURSOR[:, 1] - r.north)) <= 20 else "off"
d["status"] = d.apply(status, axis=1); print(d.status.value_counts().to_dict())
mad = lambda a, b: np.abs(T[a] - T[b]).mean()
keep = []; acc_unc = rej = 0
for farm, g in d.groupby("farm"):
    trusted = list(g.i[g.status.isin(["centre", "cursor"])])
    if not trusted: rej += len(g); continue
    unc = [i for i in g.i[~g.status.isin(["centre", "cursor"])]]
    ok = [i for i in unc if min(mad(i, j) for j in trusted) < 3]; acc_unc += len(ok); rej += len(unc) - len(ok)
    groups = []
    for i in sorted(trusted + ok):
        for gr in groups:
            if mad(i, gr[0]) < 3: gr.append(i); break
        else: groups.append([i])
    for gr in groups:
        rep = [i for i in gr if d.loc[i, "status"] == "centre"] or gr
        dates = d.loc[gr, "date"].dropna()
        keep.append((d.loc[rep[0], "file"], farm, dates.mode().iloc[0] if len(dates) else "unknown", len(gr)))
k = pd.DataFrame(keep, columns=["file", "farm", "img_date", "n_shots"])
dup = k.groupby(["farm", "img_date"]).cumcount(); k.loc[dup > 0, "img_date"] = k.img_date + "_" + dup.astype(str)
print("uncertain accepted", acc_unc, "rejected", rej)
print(len(k), "images,", k.farm.nunique(), "farms of 363;", k.groupby("farm").size().value_counts().sort_index().to_dict())
print("year>=2021:", (k.img_date.str[:4] >= "2021").sum(), "images", k[k.img_date.str[:4] >= "2021"].farm.nunique(), "farms")
k.to_csv("keep3.csv", index=False)
