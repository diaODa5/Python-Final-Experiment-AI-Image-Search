# VistaMatch — AI Image Search (Top-50) with Similarity Tiers + Web UI

VistaMatch is an end-to-end image-to-image retrieval project powered by DINOv2 features.
Upload a query image → extract embedding → search Top-50 from a gallery → group results by similarity tiers (Perfect / Excellent / Good / Just-so-so) → view everything in a clean interactive web UI.
The system also supports deduplication and search history persistence (JSON).

## ✨ Features

✅ Top-50 image retrieval using DINOv2 feature embeddings

✅ Two-stage display: results grouped by similarity tiers (Perfect / Excellent / Good / Just-so-so)

✅ Deduplication to avoid repeated images in results 
    
filename-level + near-duplicate feature filtering

✅ Web UI with smooth animations, preview modal, and rich stats

✅ Search history stored in a local search_history.json (viewable & replayable)

✅ Works on a local gallery of ~10k+ images

## 📁 Project Structure

```
├── app.py                              # FastAPI backend (search + tiers + history)

├── static/

├── index.html                      # Frontend UI

├── gallery_images/                     # Crawled gallery images (e.g., ~12k)

├── gallery_features.npy                # Extracted gallery embeddings (Nx768)

├── gallery_mapping.npy                 # Mapping from embedding row -> image filename

├── vit-dinov2-base.npz                 # DINOv2 weights (npz format)

├── build_gallery_features.py           # Build gallery_features.npy + mapping

├── search_top50.py                     # CLI topK retrieval (debug/testing)

├── preprocess_image.py                 # Image loading + resize_short_side

├── dinov2_numpy.py                     # DINOv2 forward in NumPy (pos-embed interpolation fixed)

├── download_gallery_success_target_fixed.py  # Gallery crawler (download until 12k successes)

├── requirements.txt                    # Dependencies

└── search_history.json                 # Auto generated search history (JSON)
```

## 🔄 Pipeline Overview
### 1) Build / Collect Gallery Images

The project assumes you already have a folder such as:

```
gallery_images/

  000001.jpg
  
  000002.jpg
  
  ...
```
  
If you need to crawl images from a large CSV dataset (millions of rows), use the provided crawler script:

·download_gallery_success_target_fixed.py

  ·Downloads images until N successful downloads are reached

  ·Uses concurrency and retries

  ·Stores images into gallery_images/

In practice, some URLs may be invalid or rate-limited, so the “attempted count” can be much larger than the number of successful downloads. This is expected.

### 2) Extract Gallery Features (Offline Index Building)

Run:

```bash
python build_gallery_features.py
```

This step:

·Iterates through gallery_images/

·Preprocesses each image (resize)

·Extracts a 768-dim DINOv2 embedding

·Saves two files:

✅ gallery_features.npy

✅ gallery_mapping.npy

These two .npy files form the “feature index” used by search.

### 3) Search Top-50 (CLI / Debug)

You can quickly validate search from terminal:

```bash
python search_top50.py --query /path/to/query.jpg --topk 50 --export_dir results_top50
```

This does:

·Extract query embedding

·Compute similarity against all gallery embeddings

·Return top-K matches

·Optionally export retrieved images to a folder

### 4) Web Application (FastAPI + Frontend)

Start the backend:

```
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open in browser:

```
http://127.0.0.1:8000
```

The UI provides:

·Upload a query image

·Adjust Top-K slider (default 50)

·Enable/disable deduplication

·View results grouped by similarity tiers

·Click any result to preview

·See runtime statistics and gallery size

·Browse & replay search history (stored in JSON)

## 🧠 Similarity Tiers (Second-stage Grouping)

Instead of clustering (like KMeans), VistaMatch groups results by similarity ranking.

After retrieval, Top-K results are sorted by similarity and split into:

```
·Perfect

·Excellent

·Good

·Just-so-so
```

This approach ensures the grouping is:

·stable

·easy to interpret

·aligned with “similarity from high to low”

## ♻️ Deduplication Logic

To avoid repeated images:

1.Filename dedup: same image name appears only once

2.Near-duplicate feature dedup: if a candidate image embedding is extremely close to a previously selected result (cosine similarity above a threshold), it will be removed

This makes the Top-50 results more diverse and useful.

## 🕘 Search History (JSON Persistence)

Every successful search is appended to:

```pgsql
search_history.json
```

Each record contains:

·timestamp

·query filename

·topK parameters

·runtime statistics

·the tiered results (so the UI can replay)

The UI can open History, click any record to instantly restore the result display.

## ✅ Final Output / Result

When everything is ready, the final system can:

·Build a feature index for 10k+ images

·Perform a single query search in sub-second / ~1 second range (depends on device)

·Display Top-50 matches grouped by similarity tiers

·Provide a clean and modern user experience  

## 🧩 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🛠 Notes / Troubleshooting

### 1) “broadcast shapes …” errors during feature extraction

This usually happens when positional embedding interpolation assumes square patch grids.
In this project, dinov2_numpy.py is updated to support non-square inputs, so gallery feature building can run smoothly.

### 2) .npy files not appearing

gallery_features.npy and gallery_mapping.npy are written only after the extraction finishes successfully.
If the process is interrupted, files may not be generated.

### 3) ModuleNotFoundError: fastapi

Make sure you installed requirements in the active venv:

```bash
pip install -r requirements.txt
```
And run with uvicorn:

```bash
uvicorn app:app --reload
```

