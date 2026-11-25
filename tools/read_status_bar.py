"""Read Google Earth Pro's status bar (imagery date and coordinates) from the bottom strip of every screenshot.

Input : strips/<file>.png (the 600 x 25 px bottom-right strip of each screenshot) and a PRIVATE table
        farms_master.csv with the requested GPS point of every farm (not in this repository).
Output: ocr2_<k>.csv rows: file, imagery date, lat, lon, distance to the requested point (m), eye altitude, elevation.
Usage : python read_status_bar.py <k> <n>   (process every n-th strip starting at k, so several copies can run in parallel)
"""
import re, sys, glob, os, csv, math
import numpy as np, pytesseract, pandas as pd
from PIL import Image
M = pd.read_csv("farms_master.csv", usecols=["Location", "survey/_Record_your_current_location_latitude", "survey/_Record_your_current_location_longitude"])
REQ = dict(zip(M.Location, zip(M["survey/_Record_your_current_location_latitude"].astype(float), M["survey/_Record_your_current_location_longitude"].astype(float))))
def dist(a, b):
    return 6371000 * math.hypot(math.radians(a[0] - b[0]), math.radians(a[1] - b[1]) * math.cos(math.radians(a[0])))
def img(f, thr, sc):
    a = np.asarray(Image.open(f).convert("L"), dtype=np.uint8)[:, :575]
    b = np.where(a >= thr, 0, 255).astype(np.uint8)
    return Image.fromarray(b).resize((b.shape[1] * sc, b.shape[0] * sc), Image.LANCZOS)
DMS = r"(\d{2})\D{1,3}(\d{2})\D{1,3}(\d{1,2}[.,]\d{2})"
def parse(t):
    t = t.replace(",", ".")
    d = re.search(r"(\d{1,2})/(\d{1,2})/(20\d\d)", t); ll = re.findall(DMS, t)
    alt = re.search(r"(?:alt|at)\D{0,4}(\d{4,5})", t); el = re.search(r"e\w{1,3}\s+(\d{3,4})\s", t)
    dd = f"{d.group(3)}-{int(d.group(1)):02d}-{int(d.group(2)):02d}" if d else ""
    if len(ll) < 2: return dd, None, alt and int(alt.group(1)), el and int(el.group(1))
    f = lambda g: int(g[0]) + int(g[1]) / 60 + float(g[2]) / 3600
    return dd, (f(ll[0]), f(ll[1])), alt and int(alt.group(1)), el and int(el.group(1))
TRIES = [(215, 4, 7), (190, 4, 7), (235, 4, 7), (170, 4, 7), (200, 3, 7), (225, 5, 7), (180, 3, 6)]
def run(f):
    farm = os.path.basename(f)[:5]; req = REQ[farm]; best = None; date = ""; alt = el = None
    for thr, sc, psm in TRIES:
        t = pytesseract.image_to_string(img(f, thr, sc), config=f"--psm {psm}")
        dd, ll, a, e = parse(t)
        date = date or dd; alt = alt or a; el = el or e
        if ll:
            dm = dist(ll, req)
            if best is None or dm < best[1]: best = (ll, dm)
            if dm < 60 and date: break
    if best is None: return [date, "", "", "", alt or "", el or ""]
    return [date, round(best[0][0], 6), round(best[0][1], 6), round(best[1], 1), alt or "", el or ""]
if __name__ == "__main__":
    k, n = int(sys.argv[1]), int(sys.argv[2]); fs = sorted(glob.glob("strips/*.png"))[k::n]
    out = f"ocr2_{k}.csv"; done = {r[0] for r in csv.reader(open(out))} if os.path.exists(out) else set()
    w = csv.writer(open(out, "a", newline=""))
    for i, f in enumerate(fs):
        b = os.path.basename(f)
        if b in done: continue
        w.writerow([b, *run(f)])
        if i % 200 == 0: open(out).close()
