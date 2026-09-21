# Audio biodiversity survey — Perch 2.0 pipeline

Detecting animal species by sound in a single Singapore study area, to assess
the biodiversity there. The corpus is 4,684 labelled clips from one recorder
(S01634) over 7 days, filed one folder per class.

The pipeline caches Perch 2.0 embeddings and logits once, then scores against
our own labels. Everything after extraction needs only numpy and scikit-learn,
so the expensive stage runs once and analysis iterates cheaply.

## Stages

| # | Command | Output | Cost |
|---|---------|--------|------|
| 1 | `python scripts/build_manifest.py` | `data/manifest.csv` — one row per clip, audio referenced in place | seconds |
| 2 | `python scripts/make_splits.py` | `data/splits.csv` — session-level train/val/test | seconds |
| 3 | `python scripts/build_taxonomy_map.py --assets <model assets>` | `config/perch_class_index.csv`, `config/perch_genus_taxonomy.csv` | ~5 min, **needs network**, already committed |
| 4 | `python scripts/extract_perch.py --model-name perch_v2_onnx --device cuda` | `data/perch/` — embeddings, target logits, top-K | ~22 min on GPU, 3.4 h on CPU |
| 5 | `python scripts/score_zeroshot.py` | per-class AP / AUC / F1 | seconds |

Stage 3's outputs are committed, so a fresh clone starts at stage 4 offline.
Add `--dry-run` to stage 4 to resolve targets and stop before inference.

## Why the class index exists

Perch v2 has a **single flat head of 14,795 classes** (14,597 scientific
binomials plus 198 FSD50K sound events). There are no genus, family or order
outputs, so an aggregate target like "any Orthoptera" has to be expanded into
its member species. Stage 3 does that once: birds come free from the model's
own `perch_v2_ebird_classes.csv` (a parallel array; `no_ebird_code` marks the
non-birds), and the remaining 1,730 non-bird genera are resolved against GBIF.

`config/perch_targets.csv` then expresses each of our 31 labels as a query:

```
Mixornis gularis     a literal class name
genus:Purana         every species in that genus
family=Cicadidae     every species in that family  (also order=, class=)
ebird:any            every class carrying an eBird code, i.e. every bird
```

Alternatives are `;`-separated and unioned; a target matching several classes
takes the max. Stage 4 verifies the committed index still aligns with the
model's class list and refuses to run if a model version bump shifted indices.

## GPU setup (Windows)

TensorFlow dropped native Windows GPU support after 2.10 and Perch's SavedModel
needs 2.20, so the GPU path goes through ONNX Runtime instead. Measured on this
laptop (6 GB card), batch 16:

| backend | windows/s | full corpus |
|---------|-----------|-------------|
| TensorFlow, CPU | 3.7 | 3.4 h |
| ONNX Runtime, CPU | 7.7 | 1.3 h |
| ONNX Runtime, CUDA | 26.7 | ~22 min |

```
pip install "onnxruntime-gpu==1.22.0"
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 \
            nvidia-cufft-cu12 nvidia-curand-cu12
```

Pin 1.22 deliberately: **onnxruntime-gpu 1.30 requires CUDA 13**, and driver
576.28 tops out at 12.9. Install the wrong one and it loads without error, then
silently runs on CPU. `extract_perch.py --device cuda` fails loudly instead of
falling back; `--device auto` falls back but says so.

Keep the batch at 16 on a 6 GB card. Larger batches are both slower (24→25.3,
32→24.1, 48→19.3 win/s) and eventually OOM in the ProtoPNet reduce-max.

Two things the setup gets wrong if you skip them:

- **cuDNN sublibraries.** cuDNN 9 pulls its own DLLs in by bare name through
  `LoadLibrary`, which searches `PATH` only, so `os.add_dll_directory` alone is
  not enough. CUDA initialises, then dies at the first convolution with
  `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`. `add_nvidia_dll_dirs()` sets both.
- **Peak normalisation.** perch_hoplite normalises every window to a peak of
  0.25 inside its wrapper. The fast path calls the ONNX session directly, so it
  replicates that arithmetic in `normalize_windows()`. Skipping it does not
  fail — it just quietly shifts every logit.

The GPU path is verified against the TensorFlow path on the same 258 windows:
identical top-1 and top-5 predictions, target-logit max|diff| 0.0054.

## TLS interception

This network intercepts TLS, and a fresh venv's bundled pip verifies against
certifi only, so installs fail with `CERTIFICATE_VERIFY_FAILED`. Bootstrap once:

```
python -m pip install --upgrade --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org pip
```

Stages 3 and 4's first run also download over `requests`, which needs the
Windows roots exported to a bundle and `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` /
`CURL_CA_BUNDLE` pointed at it.

## Known caveats

These belong in the write-up, not just here:

- **Single-label corpus.** One folder per clip, so a clip filed under Bird may
  also contain cicadas. Some counted false positives are annotation gaps, and
  reported precision is a lower bound.
- **Clip-level max aggregation favours long clips** — the max of n noisy window
  scores grows with n, and class correlates with duration. `score_zeroshot.py`
  reports an `auc_dur` control column: a class whose AUC is no better than
  duration alone has not been detected.
- **Scops owl ambiguity.** Perch carries *Otus lettia* (Collared) but Singapore's
  resident is *O. lempiji* (Sunda). Both are queried; 515 clips are affected.
- **`unspecified_bird` is a max over 10,222 classes** and is noisy by
  construction. This is the clearest case for a linear probe on the cached
  embeddings rather than zero-shot logits.
- **`helicopter` has no class in FSD50K**, so it falls back to `Aircraft` and is
  a near-duplicate of the `aircraft` target.
- **`Purana ubina` is absent** from Perch; three congeners stand in. Treat a hit
  as *Purana* sp., not a species ID.
- **Weather data is misaligned** — station S07, daily, full-year 2025, against
  7 days of audio. Not usable as a covariate as it stands.
