"""Part 3 data: ground photos from the Fezzan assessment.
 - survey-labelled photos kept after the Assignment 2 screening (fezzan_crops_A2, 3 classes)
 - photos hand-labelled by Omar from the left-out photos (fezzan_review/_SORTED, 4 classes)
Each photo is linked to its satellite farm ID (F0001-F1559) when there is one, and placed in a location group
(photos and satellite farms within 600 m, chained). Groups that contain a satellite TEST farm form the photo test set,
so the photo test and the satellite test cover the same places.
Writes data/photos/<source>/<label>/<file>.jpg, data/photos_index.csv (no GPS) and data/photos_index_PRIVATE.csv."""
import os, shutil, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# Private inputs (not in the repository): unpacked photo folders, the request/key workbooks and the photo GPS file
SRC = "private/photos_src"; O = "private/"; GPS = "private/sorted_gps.csv"
CLASSES = ["alfalfa", "cereals", "palm", "palm_orchard"]; SEED = 42

# ---- survey-labelled photos that passed the Assignment 2 screening
m = pd.read_csv(f"{SRC}/fezzan_crops_A2/metadata.csv"); scr = set(pd.read_csv(f"{SRC}/fezzan_crops_A2/screening.csv").file_name)
a = m[~m.file_name.isin(scr)].copy(); a["label"] = a.file_name.str.split("/").str[0]; a["photo"] = a.file_name.str.split("/").str[1]
a["path"] = SRC + "/fezzan_crops_A2/" + a.file_name; a["source"] = "survey"
k1 = pd.read_excel(O + "Fezzan_satellite_key_PRIVATE.xlsx", header=1); a["sat_farm"] = a.farm_id.map(dict(zip(k1["Survey record _id"], k1.Location)))

# ---- hand-labelled photos
h = []
for c in CLASSES:
    for f in sorted(os.listdir(f"{SRC}/hand/{c}")): h.append((f, c, f"{SRC}/hand/{c}/{f}"))
h = pd.DataFrame(h, columns=["photo", "label", "path"]); h["source"] = "hand"
g = pd.read_csv(GPS).drop_duplicates("photo"); h = h.merge(g[["photo", "lat", "lon", "uuid"]], on="photo", how="left")
k2 = pd.read_excel(O + "Fezzan_satellite_key2_PRIVATE.xlsx", header=1)
ph2loc = {p: r.Location for r in k2.itertuples() for p in str(r._10).split()}
h["sat_farm"] = h.photo.map(ph2loc)
d = pd.concat([a[["photo", "label", "path", "source", "lat", "lon", "sat_farm"]], h[["photo", "label", "path", "source", "lat", "lon", "sat_farm"]]], ignore_index=True)
print("photos:", len(d), d.groupby(["source", "label"]).size().to_dict()); print("no GPS:", d.lat.isna().sum())

# ---- near-duplicate photos (same scene taken twice): 32x32 grey thumbnails, mean abs difference < 4
T = np.stack([np.asarray(Image.open(p).convert("L").resize((32, 32)), dtype=float) for p in d.path])
dup = np.zeros(len(d), bool)
for i in range(len(d)):
    if dup[i]: continue
    j = np.where(np.abs(T[i + 1:] - T[i]).mean((1, 2)) < 4)[0] + i + 1; dup[j] = True
print("near-duplicates dropped:", dup.sum()); d = d[~dup].reset_index(drop=True)

# ---- satellite part-1 split (same code and seed as the satellite notebook)
idx = pd.read_csv("data/chips_index.csv"); idx = idx[idx.year >= 2021]
idx = idx[~idx.farm.isin(set(pd.read_csv("data/conflicting_farms.csv").farm))].reset_index(drop=True); idx["y"] = idx.label.map(CLASSES.index)
farms = idx.drop_duplicates("farm")[["farm", "y", "cluster", "area"]].reset_index(drop=True)
dev_i, test_i = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(farms, farms.y, farms.cluster))
sat_test = set(farms.farm[test_i])

# ---- location groups: photos + satellite farms, 600 m chained
r = pd.concat([pd.read_excel(O + "Fezzan_satellite_requests.xlsx", sheet_name="Requests"), pd.read_excel(O + "Fezzan_satellite_requests_batch3.xlsx", sheet_name="Requests")])
pts = pd.concat([pd.DataFrame({"key": "S:" + r.Location, "lat": r.Latitude, "lon": r.Longitude}),
                 pd.DataFrame({"key": "P:" + d.index.astype(str), "lat": d.lat, "lon": d.lon})], ignore_index=True)
la, lo = np.radians(pts.lat.values), np.radians(pts.lon.values)
D = 2 * 6371000 * np.arcsin(np.sqrt(np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2))
par = list(range(len(pts)))
def find(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
for i, j in zip(*np.where(np.triu(D < 600, 1))): par[find(i)] = find(j)
pts["comp"] = [find(i) for i in range(len(pts))]
test_comp = set(pts.comp[pts.key.isin({"S:" + f for f in sat_test})])
pc = pts[pts.key.str.startswith("P:")].set_index("key").comp
d["group"] = [int(pc[f"P:{i}"]) for i in d.index]
d["split"] = np.where(d.group.isin(test_comp), "test", "dev")
d["sat_test_farm"] = d.sat_farm.isin(sat_test)
print(pd.crosstab([d.source, d.label], d.split, margins=True))
print("photos of satellite test farms:", d.sat_test_farm.sum(), "| photos linked to any satellite farm:", d.sat_farm.notna().sum())

# ---- copy and write index
d["image"] = "photos/" + d.source + "/" + d.label + "/" + d.photo
for s, p in zip(d.image, d.path):
    os.makedirs(os.path.dirname(f"data/{s}"), exist_ok=True); shutil.copy(p, f"data/{s}")
d.drop(columns=["path"]).to_csv("data/photos_index_PRIVATE.csv", index=False)
d[["image", "label", "source", "sat_farm", "group", "split"]].to_csv("data/photos_index.csv", index=False)
print("saved", len(d))
