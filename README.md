# Crop type from satellite views and ground photos: Fezzan, Libya


TME-6015 final project (UNB). Code for classifying the main crop group of a farm (alfalfa, cereals, palm, palm orchard) from
(1) a Google Earth view around the farm's GPS point and (2) the enumerator's ground photo, using records from the
2023 Fezzan Agriculture Assessment. The predictions are used to flag survey records whose reported crop is unlikely,
so the quality control team can check those first.

## Results (test set)

| Model | Input | Accuracy | Macro-F1 | Macro AUC |
|---|---|---|---|---|
| Majority class baseline | satellite | 0.331 | 0.124 | 0.50 |
| CNN from scratch (S1) | satellite | 0.421 | 0.283 | 0.54 |
| DenseNet121 frozen + dense head (T5) | satellite | 0.396 | 0.351 | 0.63 |
| Majority class baseline | photo | 0.247 | 0.099 | 0.50 |
| CNN from scratch (P0) | photo | 0.432 | 0.426 | 0.72 |
| ResNet50 frozen + logistic regression, mirrored (P4) | photo | 0.574 | 0.577 | 0.81 |

Same 138 test farms with both a photo and a satellite view: photo model macro-F1 0.570, satellite model 0.276, average of both 0.534.
Full tables: `outputs/test_summary.csv`, `outputs/photo_test_summary.csv`, `outputs/photo_vs_satellite.csv`.

## Repository structure

```
fezzan_satellite_crops.ipynb                 part 1: satellite views, CNN from scratch vs frozen backbones (executed)
fezzan_satellite_crops_part2_batch3.ipynb    part 2: does adding 264 hand-labelled farms help? (executed)
fezzan_photos_part3.ipynb                    part 3: ground photos, and photo vs satellite on the same farms (executed)
predict.py                                   satellite inference and survey check
predict_photo.py                             photo inference and reported-crop check
data/                                        indexes and labels (no images, no GPS)
outputs/                                     result tables, figures and trained models
tools/                                       dataset creation scripts and the notebook builders
```

## Setup

Python 3.10 or newer.
```
pip install -r requirements.txt
```
The notebooks load ImageNet weights through Keras (downloaded automatically on first use). If the download is blocked,
put the `*_notop.h5` weight files in `weights/`.

## Data

The images are included in this version, in the layout the notebooks expect:
```
data/chips/<farm>/<farm>_<image date>.jpg          2,781 satellite views, 512 x 512 px (about 580 m wide)
data/photos/<source>/<label>/<photo>.jpg           557 ground photos (source = survey or hand)
```
Index files in `data/`:
- `chips_index.csv`: part 1 satellite views (image, farm ID, label, image date, spatial cluster, area code)
- `chips_index_batch3.csv`: batch 3 satellite views (labels from the hand-sorted photos)
- `photos_index.csv`: ground photos (label, label source, linked satellite farm, location group, split)
- `conflicting_farms.csv`: farm pairs with the same view but different reported crops (excluded)

Farm IDs (F0001-F1559) are anonymous. The key linking them to survey records is kept private.

## Training and evaluation

Run in order (each notebook writes to `outputs/`):
```
jupyter nbconvert --to notebook --execute fezzan_satellite_crops.ipynb --output fezzan_satellite_crops_executed.ipynb
jupyter nbconvert --to notebook --execute fezzan_satellite_crops_part2_batch3.ipynb --output part2_executed.ipynb
jupyter nbconvert --to notebook --execute fezzan_photos_part3.ipynb --output part3_executed.ipynb
```
Quick test on a small subset: set the environment variable `FAST=1` first (results go to `outputs_smoke/`).
On Google Colab: upload the repository and the images, `%cd` into the folder and run the notebooks cell by cell.

Every experiment changes one thing, scored by pooled out-of-fold macro-F1 over 5 development folds grouped by location
(farms within 600 m stay together). The test set is used once, after all choices.

## Inference

Satellite view(s) of one farm:
```
python predict.py data/chips/F0002/F0002_2021-08-26.jpg
```
Survey check for every farm in an index file (writes a review list sorted by how unlikely the reported crop is):
```
python predict.py --check data/chips_index.csv --out review_list.csv
```
Ground photo, with a check of the reported crop:
```
python predict_photo.py photo.jpg --reported cereals
```
A record is flagged when the reported crop has probability below 0.15 and another crop above 0.60.

## Dataset creation (tools/)

1. `read_status_bar.py`: read the real imagery date and position from each Google Earth Pro screenshot's status bar (OCR).
2. `select_views.py`: keep verified screenshots only, one copy per distinct image.
3. `crop_views.py`: cut a 512 x 512 px view around each farm.
4. `select_views_batch3.py`, `prep_batch3.py`: the same steps for batch 3, and spatial links to the part-1 clusters.
5. `prep_photos.py`: photo index, near-duplicate removal, location groups shared with the satellite test set.
6. `make_notebook*.py`: build the three notebooks.

Scripts 1, 4 and 5 need the private files (survey key, GPS points) and are included to document the steps.

## Data availability

- Survey data and ground photos: Fezzan Agriculture Assessment (2023), carried out by Tatweer Research for WFP Libya.
  Not public. This private version includes the photos; the public version shares only anonymous indexes.
- Satellite views: Google Earth Pro screenshots (imagery © Google and its data providers). Google's terms do not allow
  redistribution, so this version must not be shared publicly. All figures are kept here; the public version removes the ones
  that show imagery or photos.
