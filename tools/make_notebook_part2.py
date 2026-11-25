"""Builds fezzan_satellite_crops_part2_batch3.ipynb: does more (noisier) training data help?
Reuses the setup, data, split and helper cells of part 1 unchanged, so the development folds and the test set are identical."""
import nbformat as nbf

P1 = nbf.read("fezzan_satellite_crops.ipynb", 4).cells
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))
def copy(i): C.append(nbf.v4.new_code_cell(P1[i].source))

md("""
# Part 2: more training data from the hand-sorted photos (batch 3)

Batch 3 = 353 more farms. These are mixed-crop farms whose ground photo was sorted by hand into one of the 4 classes; their Google Earth views were collected and checked with the same steps as part 1 (status-bar reading, thumbnail match, one copy per distinct image).

Question: does adding these farms to the TRAINING folds improve the models? Scoring is on exactly the same development farms and folds as part 1, so the numbers are comparable. Batch 3 farms are never used for validation or testing.

Rule as in part 1: one change per model; a change is kept only if the pooled out-of-fold macro-F1 goes up.
""")
md("## 0. Setup (same as part 1)")
copy(2)
md("## 1. Data and splits (same code as part 1, so the folds and the test set are identical)")
copy(4); copy(5)
code("""
def load(path, ctx, px):
    im = Image.open(f"{DATA}/{path}").convert("RGB")
    if ctx == 290: im = im.crop((128, 128, 384, 384))
    return np.asarray(im.resize((px, px), Image.BILINEAR), dtype=np.uint8)
X224 = {580: np.stack([load(p, 580, 224) for p in idx.image])}; X128 = {580: np.stack([load(p, 580, 128) for p in idx.image])}
cfg = json.load(open(f"{OUT}/models/config.json")); cnn_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg["cnn"].items()}; tl = cfg["transfer"]
print("part 1 final settings -> CNN from scratch:", cnn_cfg, "| transfer:", tl)
""")
md("## 2. Batch 3 and which folds may use it")
code("""
b3 = pd.read_csv(f"{DATA}/chips_index_batch3.csv", dtype={"link_clusters": str})
print("batch 3 images:", len(b3), "| farms:", b3.farm.nunique())
b3 = b3[b3.year >= 2021].reset_index(drop=True); b3["y"] = b3.label.map(CLASSES.index)

# A batch 3 farm within 600 m (chained) of a part-1 farm shares that farm's spatial cluster.
# - linked to a TEST cluster -> not used at all (it would leak the test areas into training)
# - linked to a dev cluster in fold k -> not used when fold k is the validation fold
test_cl = set(idx.cluster[idx.split == "test"]); fold_cl = idx[idx.split == "dev"].groupby("cluster").fold.first().to_dict()
links = b3.link_clusters.fillna("").astype(str).str.split()
b3["to_test"] = [any(int(c) in test_cl for c in l) for l in links]
b3["blocked"] = [sorted({fold_cl[int(c)] for c in l if int(c) in fold_cl}) for l in links]
b3 = b3[~b3.to_test].reset_index(drop=True)
print("dropped (linked to test clusters):", int(links.map(lambda l: any(int(c) in test_cl for c in l)).sum()), "images")
print("used:", len(b3), "images |", b3.farm.nunique(), "farms | farms whose label disagrees with the survey crops:", b3.drop_duplicates("farm").survey_conflict.sum())
display(pd.crosstab(b3.label, b3.blocked.map(lambda b: "blocked in fold " + ",".join(str(k + 1) for k in b) if b else "usable in every fold")))
print("batch 3 farms per class:", b3.drop_duplicates("farm").label.value_counts().to_dict())
print("part-1 dev farms per class:", idx[idx.split == "dev"].drop_duplicates("farm").label.value_counts().to_dict())

Xb224 = np.stack([load(p, 580, 224) for p in b3.image]); Xb128 = np.stack([load(p, 580, 128) for p in b3.image]); yb = b3.y.to_numpy(dtype=np.int64)
def b3_rows(k, drop_conflict=False):
    \"\"\"Batch 3 rows allowed in training when fold k is the validation fold (k = -1: final model, all folds).\"\"\"
    ok = np.array([k not in b for b in b3.blocked]) if k >= 0 else np.ones(len(b3), bool)
    if drop_conflict: ok &= b3.survey_conflict.values == 0
    return np.where(ok)[0]

fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
for r, c in enumerate(CLASSES):
    for j, i in enumerate(b3[b3.label == c].sample(6, random_state=1).index):
        ax[r, j].imshow(Xb224[i]); ax[r, j].set_title(f"batch 3 · {c}", fontsize=9); ax[r, j].axis("off")
plt.suptitle("Six random batch 3 views per class (label from the hand-sorted photo)"); plt.tight_layout(); plt.savefig(f"{OUT}/samples_batch3.png", dpi=110); plt.show()
""")
md("## 3. Helpers (same as part 1) and training functions that can add batch 3")
copy(8); copy(10); copy(16)
code("""
def dense_head(d):
    keras.utils.set_random_seed(SEED)
    inp = keras.Input((d,)); x = layers.Dense(256, activation="relu")(inp); x = layers.Dropout(0.4)(x)
    return keras.Model(inp, layers.Dense(N, activation="softmax")(x))

def features_b3(name, ctx=580):
    path = f"{OUT}/features_batch3_{name}_{ctx}.npy"
    if os.path.exists(path) and np.load(path).shape[1] == len(b3): return np.load(path)
    m, pre = backbone(name); F = np.stack([m.predict(pre(np.rot90(Xb224, r, axes=(1, 2)).astype("float32")), batch_size=32, verbose=0) for r in range(4)])
    np.save(path, F); return F

def run_cnn2(name, change, use_b3=False, drop_conflict=False, max_epochs=30, patience=6, lr=1e-3, **kw):
    max_epochs = 2 if FAST else max_epochs; X = X128[580]; P = np.zeros((len(DEV), N)); epochs = []; t = time.time(); n_extra = []
    for k in range(5):
        tr = DEV[idx.fold.values[DEV] != k]; va = DEV[idx.fold.values[DEV] == k]
        Xtr, ytr = X[tr], y[tr]
        if use_b3: e = b3_rows(k, drop_conflict); Xtr, ytr = np.concatenate([Xtr, Xb128[e]]), np.concatenate([ytr, yb[e]]); n_extra.append(len(e))
        m = build_cnn(**kw); m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True)]
        if k == 0: pcb = PerClass(X[va], y[va]); cb.append(pcb)
        h = m.fit(make_ds(Xtr, ytr, True), validation_data=make_ds(X[va], y[va], False), epochs=max_epochs, callbacks=cb, verbose=0).history
        if k == 0: HIST[name] = {**h, "per_class": pcb.v}
        P[np.searchsorted(DEV, va)] = m.predict(X[va], verbose=0); epochs.append(int(np.argmin(h["val_loss"])) + 1)
    row = score(name, change, P, {"weights": m.count_params(), "best epochs": epochs, "batch 3 images per fold": n_extra, "time (s)": round(time.time() - t)})
    plot_curves(name); return row

def run_dense2(name, change, use_b3=False, drop_conflict=False, max_epochs=100, patience=10):
    max_epochs = 2 if FAST else max_epochs; F = features(tl["backbone"], tl["ctx"]); Fb = features_b3(tl["backbone"], tl["ctx"])
    P = np.zeros((len(DEV), N)); t = time.time(); epochs = []; n_extra = []
    mu, sd = F[0][DEV].mean(0), F[0][DEV].std(0) + 1e-6; Fz, Fbz = (F - mu) / sd, (Fb - mu) / sd
    R = range(4) if tl["rotations"] else range(1)
    for k in range(5):
        tr = DEV[idx.fold.values[DEV] != k]; va = DEV[idx.fold.values[DEV] == k]
        Xtr, ytr = np.concatenate([Fz[r][tr] for r in R]), np.tile(y[tr], len(R))
        if use_b3:
            e = b3_rows(k, drop_conflict); n_extra.append(len(e))
            Xtr, ytr = np.concatenate([Xtr] + [Fbz[r][e] for r in R]), np.concatenate([ytr, np.tile(yb[e], len(R))])
        m = dense_head(F.shape[2]); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True)]
        if k == 0: pcb = PerClass(Fz[0][va], y[va]); cb.append(pcb)
        h = m.fit(Xtr, ytr, validation_data=(Fz[0][va], y[va]), epochs=max_epochs, batch_size=64, callbacks=cb, verbose=0).history
        if k == 0: HIST[name] = {**h, "per_class": pcb.v}
        P[np.searchsorted(DEV, va)] = np.mean([m.predict(Fz[r][va], verbose=0) for r in R], axis=0); epochs.append(int(np.argmin(h["val_loss"])) + 1)
    row = score(name, change, P, {"weights": m.count_params(), "best epochs": epochs, "batch 3 images per fold": n_extra, "time (s)": round(time.time() - t)})
    plot_curves(name); return row
""")
md("## 4. Experiments (one change each)")
code("""
# Reference rows: the part-1 final models, re-run here so every number in this table comes from the same session
s1 = run_cnn2("S1", "part-1 final CNN from scratch (3 blocks, rotations/flips)", **cnn_cfg)
t5 = run_dense2("T5", "part-1 final transfer model (DenseNet121, 580 m, dense head)")
""")
code("""
s4 = run_cnn2("S4", "S1 + batch 3 farms in the training folds", use_b3=True, **cnn_cfg)
best_s, best_s_row, best_s_kw = ("S4", s4, dict(use_b3=True)) if better(s4, s1) else ("S1", s1, dict())
print("kept:", best_s)
""")
code("""
t6 = run_dense2("T6", "T5 + batch 3 farms in the training folds", use_b3=True)
best_t, best_t_row, best_t_kw = ("T6", t6, dict(use_b3=True)) if better(t6, t5) else ("T5", t5, dict())
print("kept:", best_t)
""")
code("""
t7 = run_dense2("T7", "T6 without the batch 3 farms whose photo label is not among the crops in the survey", use_b3=True, drop_conflict=True)
if better(t7, best_t_row): best_t, best_t_row, best_t_kw = "T7", t7, dict(use_b3=True, drop_conflict=True)
print("final transfer model:", best_t, best_t_kw)
display(table()); table().to_csv(f"{OUT}/experiments_part2.csv")
""")
code("""
# Per-class recall, part-1 final models vs the batch 3 versions (development set, out-of-fold)
cols = [f"recall {c}" for c in CLASSES]
display(table()[cols + ["val macro-F1"]])
fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.8 / len(RESULTS)
for j, r in enumerate(RESULTS): ax.bar(np.arange(N) + j * w, [r[c] for c in cols], w, label=r["model"])
ax.set_xticks(np.arange(N) + 0.4 - w / 2); ax.set_xticklabels(CLASSES); ax.set_ylabel("recall (dev, out-of-fold)"); ax.set_ylim(0, 1); ax.legend(ncol=5, fontsize=8)
plt.title("Recall per class with and without batch 3"); plt.tight_layout(); plt.savefig(f"{OUT}/recall_part2.png", dpi=110); plt.show()
""")
md("""
## 5. Test set (second use)

The test set was already used once in part 1. It is used here a second time, only to compare the part-1 models with the batch 3 models chosen above (the choice was made on the development folds only). Same test farms, same images.
""")
code("""
def fit_cnn_final(use_b3=False, drop_conflict=False):
    row = s4 if use_b3 else s1; ep = int(np.median(row["best epochs"]))
    Xtr, ytr = X128[580][DEV], y[DEV]
    if use_b3: e = b3_rows(-1, drop_conflict); Xtr, ytr = np.concatenate([Xtr, Xb128[e]]), np.concatenate([ytr, yb[e]])
    m = build_cnn(**cnn_cfg); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(make_ds(Xtr, ytr, True), epochs=ep, verbose=0); return m

def fit_dense_final(row, use_b3=False, drop_conflict=False):
    F = features(tl["backbone"], tl["ctx"]); Fb = features_b3(tl["backbone"], tl["ctx"]); mu, sd = F[0][DEV].mean(0), F[0][DEV].std(0) + 1e-6
    Xtr, ytr = (F[0][DEV] - mu) / sd, y[DEV]
    if use_b3: e = b3_rows(-1, drop_conflict); Xtr, ytr = np.concatenate([Xtr, (Fb[0][e] - mu) / sd]), np.concatenate([ytr, yb[e]])
    m = dense_head(F.shape[2]); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(Xtr, ytr, epochs=int(np.median(row["best epochs"])), batch_size=64, verbose=0)
    return m, mu, sd, m.predict((F[0][TEST] - mu) / sd, verbose=0)

models = {}
cnn_old = keras.models.load_model(f"{OUT}/models/cnn_scratch.keras", custom_objects={"RotFlip": RotFlip}); models["CNN from scratch, part 1 (S1)"] = cnn_old.predict(X128[580][TEST], verbose=0)
if best_s != "S1": cnn_new = fit_cnn_final(**best_s_kw); models[f"CNN from scratch + batch 3 ({best_s})"] = cnn_new.predict(X128[580][TEST], verbose=0)
_, _, _, models["transfer, part 1 (T5)"] = fit_dense_final(t5)
if best_t != "T5": head_new, mu, sd, models[f"transfer + batch 3 ({best_t})"] = fit_dense_final(best_t_row, **best_t_kw)

yt = y[TEST]; summary = {}
for nm, P in models.items():
    fy, fp = farm_level(TEST, P)
    summary[nm] = {"accuracy": accuracy_score(yt, P.argmax(1)), "macro-F1": f1_score(yt, P.argmax(1), average="macro"),
                   "farm-level macro-F1": f1_score(fy, fp.argmax(1), average="macro"), "macro AUC (one-vs-rest)": roc_auc_score(yt, P, multi_class="ovr", average="macro"),
                   **{f"recall {c}": r for c, r in zip(CLASSES, per_class_recall(yt, P.argmax(1)))}}
summary = pd.DataFrame(summary).round(3); display(summary); summary.to_csv(f"{OUT}/test_summary_part2.csv")
print("part 1 test summary (saved file, for reference):"); display(pd.read_csv(f"{OUT}/test_summary.csv", index_col=0))
""")
code("""
names = [n for n in models if "batch 3" in n]
if names:
    fig, ax = plt.subplots(1, len(names), figsize=(6.5 * len(names), 5.2), squeeze=False)
    for a, nm in zip(ax[0], names):
        ConfusionMatrixDisplay(confusion_matrix(yt, models[nm].argmax(1), labels=range(N)), display_labels=CLASSES).plot(ax=a, cmap="Greens", colorbar=False)
        a.set_title(f"{nm}, test images", fontsize=10); a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(f"{OUT}/test_confusion_part2.png", dpi=110); plt.show()

# Bootstrap over test farms: farm-level macro-F1 of each batch 3 model minus its part-1 version
rng = np.random.default_rng(SEED)
for new, old in [(n, "CNN from scratch, part 1 (S1)" if "CNN" in n else "transfer, part 1 (T5)") for n in names]:
    fy, f_new = farm_level(TEST, models[new]); _, f_old = farm_level(TEST, models[old]); d = []
    for _ in range(3000):
        s = rng.integers(0, len(fy), len(fy)); d.append(f1_score(fy[s], f_new[s].argmax(1), average="macro") - f1_score(fy[s], f_old[s].argmax(1), average="macro"))
    d = np.array(d); print(f"{new} minus {old}: {np.mean(d):+.3f}, 95% interval {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}, ahead in {np.mean(d > 0):.1%} of resamples")
if not names: print("Neither batch 3 model beat its part-1 version on the development folds, so the test set was not used again.")
""")
md("## 6. How well do the part-1 models agree with the batch 3 photo labels?")
code("""
# Part-1 transfer model (trained on development farms only) applied to the batch 3 views: a check on the hand labels of mixed farms.
F = features(tl["backbone"], tl["ctx"]); Fb = features_b3(tl["backbone"], tl["ctx"]); mu1, sd1 = F[0][DEV].mean(0), F[0][DEV].std(0) + 1e-6
h1 = dense_head(F.shape[2]); h1.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h1.fit((F[0][DEV] - mu1) / sd1, y[DEV], epochs=int(np.median(t5["best epochs"])), batch_size=64, verbose=0)
Pb = h1.predict((Fb[0] - mu1) / sd1, verbose=0)
df = pd.DataFrame(Pb, columns=CLASSES); df["farm"] = b3.farm; df["y"] = yb; g = df.groupby("farm").mean(numeric_only=True)
fy, fp = g.y.astype(int).values, g[CLASSES].values
print(f"batch 3 farms: {len(fy)} | agreement with the photo label: accuracy {accuracy_score(fy, fp.argmax(1)):.3f}, macro-F1 {f1_score(fy, fp.argmax(1), average='macro'):.3f}")
print("recall per class:", {c: round(r, 2) for c, r in zip(CLASSES, per_class_recall(fy, fp.argmax(1)))})
ConfusionMatrixDisplay(confusion_matrix(fy, fp.argmax(1), labels=range(N)), display_labels=CLASSES).plot(cmap="Greens", colorbar=False)
plt.title("Part-1 transfer model on batch 3 farms (rows = photo label)"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(f"{OUT}/batch3_agreement.png", dpi=110); plt.show()
""")
md("## 7. Save the batch 3 models (only if they were kept)")
code("""
os.makedirs(f"{OUT}/models_part2", exist_ok=True)
if best_s != "S1": cnn_new.save(f"{OUT}/models_part2/cnn_scratch.keras")
if best_t != "T5": head_new.save(f"{OUT}/models_part2/transfer_head.keras"); np.savez(f"{OUT}/models_part2/transfer_scaler.npz", mu=mu, sd=sd)
json.dump({"classes": CLASSES, "cnn": {"model": best_s, **{k: list(v) if isinstance(v, tuple) else v for k, v in cnn_cfg.items()}},
           "transfer": {**tl, "model": best_t, **best_t_kw}}, open(f"{OUT}/models_part2/config.json", "w"), indent=1)
print("kept models:", best_s, best_t)
""")

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "fezzan_satellite_crops_part2_batch3.ipynb"); print("cells:", len(C))
