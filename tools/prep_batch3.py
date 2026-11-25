"""Index for batch 3 (farms behind the hand-sorted leftover photos). Writes data/chips_index_batch3.csv (public, no GPS)
and data/chips_index_batch3_PRIVATE.csv. link_clusters = clusters of the first 1,162 farms that are within 600 m (chained)."""
import numpy as np, pandas as pd
k = pd.read_csv("keep3.csv")
m = pd.read_csv("farms_master_batch3_PRIVATE.csv")
NEED = {"alfalfa": "fodder", "cereals": "cereals", "palm": "palm", "palm_orchard": "palm"}
m["survey_conflict"] = [int(c not in ("none reported", "not in survey export") and NEED[l] not in c) for l, c in zip(m.label, m.survey_crops)]
old = pd.read_csv("data/chips_index_PRIVATE.csv").drop_duplicates("farm")[["farm", "lat", "lon", "cluster"]]
areas = pd.read_csv("data/area_codes_PRIVATE.csv"); amap = dict(zip(areas.area_name, areas.area))
for r in sorted(set(m.region) - set(amap)): amap[r] = f"A{len(amap) + 1:02d}"
pd.DataFrame({"area_name": list(amap), "area": list(amap.values())}).to_csv("data/area_codes_PRIVATE.csv", index=False)

# union-find over old + new farms, edge if < 600 m
P = pd.concat([old.assign(new=0), m.rename(columns={"Location": "farm"})[["farm", "lat", "lon"]].assign(new=1, cluster=-1)], ignore_index=True)
la, lo = np.radians(P.lat.values), np.radians(P.lon.values)
D = 2 * 6371000 * np.arcsin(np.sqrt(np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2))
par = list(range(len(P)))
def find(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
for i, j in zip(*np.where(np.triu(D < 600, 1))): par[find(i)] = find(j)
P["comp"] = [find(i) for i in range(len(P))]
links = P[P.new == 0].groupby("comp").cluster.apply(lambda s: " ".join(map(str, sorted(set(s)))))
newc = {c: 10000 + i for i, c in enumerate(sorted(set(P.comp[P.new == 1])))}
P["link_clusters"] = P.comp.map(links).fillna(""); P["cluster3"] = P.comp.map(newc)
info = m.rename(columns={"Location": "farm"}).merge(P[P.new == 1][["farm", "link_clusters", "cluster3"]], on="farm")
info["area"] = info.region.map(amap)
x = k.merge(info[["farm", "label", "lat", "lon", "region", "area", "survey_conflict", "link_clusters", "cluster3", "farm_size"]], on="farm")
x["image"] = "chips/" + x.farm + "/" + x.farm + "_" + x.img_date + ".jpg"
x["year"] = pd.to_numeric(x.img_date.str[:4], errors="coerce")
x = x.rename(columns={"cluster3": "cluster"}); x["batch"] = 3
x.to_csv("data/chips_index_batch3_PRIVATE.csv", index=False)
x[["image", "farm", "label", "img_date", "year", "n_shots", "cluster", "link_clusters", "area", "survey_conflict", "batch"]].to_csv("data/chips_index_batch3.csv", index=False)
f = x.drop_duplicates("farm")
print(len(x), "images", len(f), "farms |", f.label.value_counts().to_dict())
print("farms linked to old clusters:", (f.link_clusters != "").sum(), "| survey conflict:", f.survey_conflict.sum())
import os; miss = [p for p in x.image if not os.path.exists("data/" + p)]; print("missing files:", len(miss))
