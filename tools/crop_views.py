"""Cut a 512 x 512 px square (about 580 m) around the view centre of each accepted Google Earth Pro screenshot.

Input : keep.csv with columns file, farm, img_date (one row per distinct image of a farm)
Output: chips512/<farm>/<farm>_<img_date>.png

The screenshots are 1920 x 991 px full-window captures. The map view spans x 391-1920, y 47-991,
and its centre (1155, 515) is the farm's GPS point. Range above ground is constant (~1,500 m),
which gives about 1.13 m per pixel.
"""
import os, csv, argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--screenshots", required=True); ap.add_argument("--keep", required=True); ap.add_argument("--out", default="chips512")
a = ap.parse_args()
CX, CY, HALF = 1155, 515, 256
for r in csv.DictReader(open(a.keep)):
    out = os.path.join(a.out, r["farm"], f'{r["farm"]}_{r["img_date"]}.png')
    if os.path.exists(out): continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.open(os.path.join(a.screenshots, r["farm"], r["file"])).convert("RGB").crop((CX - HALF, CY - HALF, CX + HALF, CY + HALF)).save(out)
