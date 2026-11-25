"""Decide which screenshots to keep, and keep one copy of each distinct image per farm.

Inputs: ocr_all.pkl (status-bar readings, from read_status_bar.py, with east/north offsets in metres to the
        requested point) and thumbs_all.npz (32 x 32 grey thumbnail of the central 512 px of every screenshot).
Rules : a screenshot is trusted when its status-bar position is within 60 m of the farm's GPS point ("centre"),
        or when the offset equals one of the fixed mouse-pointer offsets seen across many farms ("cursor").
        Other screenshots are accepted only if their thumbnail matches a trusted screenshot of the same farm
        (mean absolute difference < 3 grey levels). Accepted screenshots with matching thumbnails are one image.
Output: keep.csv (file, farm, img_date, n_shots)
"""
import numpy as np, pandas as pd

d = pd.read_pickle("ocr_all.pkl")
z = np.load("thumbs_all.npz"); T = z["T"].astype(float); F = [f.split("/")[1] for f in z["files"]]
d = d.set_index("file").loc[F].reset_index(); d["i"] = range(len(d))
CURSOR = np.array([(-125, -445), (155, -470), (166, -301)])     # metres east, north

def status(r):
    if pd.isna(r.dist): return "unread"
    if r.dist <= 60: return "centre"
    return "cursor" if np.min(np.hypot(CURSOR[:, 0] - r.east, CURSOR[:, 1] - r.north)) <= 15 else "off"

d["status"] = d.apply(status, axis=1)
mad = lambda a, b: np.abs(T[a] - T[b]).mean()
keep = []
for farm, g in d.groupby("farm"):
    trusted = list(g.i[g.status.isin(["centre", "cursor"])])
    if not trusted: continue
    acc = trusted + [i for i in g.i[~g.status.isin(["centre", "cursor"])] if min(mad(i, j) for j in trusted) < 3]
    groups = []
    for i in sorted(acc):
        for gr in groups:
            if mad(i, gr[0]) < 3: gr.append(i); break
        else: groups.append([i])
    for gr in groups:
        rep = [i for i in gr if d.loc[i, "status"] == "centre"] or gr
        keep.append((d.loc[rep[0], "file"], farm, d.loc[gr, "date"].mode().iloc[0], len(gr)))
k = pd.DataFrame(keep, columns=["file", "farm", "img_date", "n_shots"])
dup = k.groupby(["farm", "img_date"]).cumcount(); k.loc[dup > 0, "img_date"] = k.img_date + "_" + dup.astype(str)
k.to_csv("keep.csv", index=False); print(len(k), "images,", k.farm.nunique(), "farms")
