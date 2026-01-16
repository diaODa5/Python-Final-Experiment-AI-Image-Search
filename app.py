import os
import time
import json
import uuid
import tempfile
from datetime import datetime
from typing import Any, Optional, List, Dict

import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# =========================
# Project
# =========================
PROJECT_NAME = "VistaMatch"

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", os.path.join(BASE_DIR, "vit-dinov2-base.npz"))
GALLERY_FEATURES_PATH = os.getenv("GALLERY_FEATURES_PATH", os.path.join(BASE_DIR, "gallery_features.npy"))
GALLERY_MAPPING_PATH = os.getenv("GALLERY_MAPPING_PATH", os.path.join(BASE_DIR, "gallery_mapping.npy"))
GALLERY_DIR = os.getenv("GALLERY_DIR", os.path.join(BASE_DIR, "gallery_images"))
STATIC_DIR = os.getenv("STATIC_DIR", os.path.join(BASE_DIR, "static"))

HISTORY_PATH = os.getenv("HISTORY_PATH", os.path.join(BASE_DIR, "search_history.json"))
HISTORY_MAX_LEN = int(os.getenv("HISTORY_MAX_LEN", "200"))  # 最多保留多少条历史

DEFAULT_TOPK = 50
DEDUP_FEAT_SIM_TH = float(os.getenv("DEDUP_FEAT_SIM_TH", "0.9995"))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.reshape(-1).astype(np.float32)
    return x / (np.linalg.norm(x) + eps)


def _pack_tier(name: str, items: List[dict]) -> dict:
    if not items:
        return {"name": name, "size": 0, "score_max": None, "score_min": None, "avg_score": None, "items": []}
    scores = [float(it["score"]) for it in items]
    return {
        "name": name,
        "size": int(len(items)),
        "score_max": float(max(scores)),
        "score_min": float(min(scores)),
        "avg_score": float(sum(scores) / len(scores)),
        "items": items,
    }


def split_into_tiers_by_quantile(items_sorted: List[dict]) -> List[dict]:
    """
    items_sorted: 按 score 从高到低
    分位数分档：
      Perfect: >= 90%
      Excellent: 70%~90%
      Good: 40%~70%
      Just-so-so: <40%
    """
    k = len(items_sorted)
    if k == 0:
        return []

    if k < 6:
        # 样本少时按数量兜底
        return split_into_tiers_by_count(items_sorted)

    scores = np.array([it["score"] for it in items_sorted], dtype=np.float32)
    q90 = float(np.quantile(scores, 0.90))
    q70 = float(np.quantile(scores, 0.70))
    q40 = float(np.quantile(scores, 0.40))

    perfect, excellent, good, soso = [], [], [], []
    for it in items_sorted:
        s = float(it["score"])
        if s >= q90:
            perfect.append(it)
        elif s >= q70:
            excellent.append(it)
        elif s >= q40:
            good.append(it)
        else:
            soso.append(it)

    non_empty = sum(1 for grp in [perfect, excellent, good, soso] if grp)
    if non_empty < 3:
        return split_into_tiers_by_count(items_sorted)

    return [
        _pack_tier("Perfect", perfect),
        _pack_tier("Excellent", excellent),
        _pack_tier("Good", good),
        _pack_tier("Just-so-so", soso),
    ]


def split_into_tiers_by_count(items_sorted: List[dict]) -> List[dict]:
    k = len(items_sorted)
    if k == 0:
        return []

    c1 = max(1, int(round(k * 0.10)))
    c2 = max(c1 + 1, int(round(k * 0.30))) if k >= 2 else 1
    c3 = max(c2 + 1, int(round(k * 0.60))) if k >= 3 else 1
    c1, c2, c3 = min(c1, k), min(c2, k), min(c3, k)

    return [
        _pack_tier("Perfect", items_sorted[:c1]),
        _pack_tier("Excellent", items_sorted[c1:c2]),
        _pack_tier("Good", items_sorted[c2:c3]),
        _pack_tier("Just-so-so", items_sorted[c3:]),
    ]


# =========================
# History JSON helpers
# =========================
def _read_history() -> List[dict]:
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_history(entry: dict) -> None:
    hist = _read_history()
    hist.insert(0, entry)  # newest first
    if len(hist) > HISTORY_MAX_LEN:
        hist = hist[:HISTORY_MAX_LEN]
    _atomic_write_json(HISTORY_PATH, hist)


# =========================
# FastAPI
# =========================
app = FastAPI(title=f"{PROJECT_NAME} · AI Image Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/gallery_images", StaticFiles(directory=GALLERY_DIR), name="gallery_images")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL: Optional[Dinov2Numpy] = None
GALLERY_NORM: Optional[np.ndarray] = None
GALLERY_NAMES: Optional[np.ndarray] = None


@app.on_event("startup")
def _startup():
    global MODEL, GALLERY_NORM, GALLERY_NAMES

    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(f"Missing weights: {WEIGHTS_PATH}")
    if not os.path.exists(GALLERY_FEATURES_PATH) or not os.path.exists(GALLERY_MAPPING_PATH):
        raise RuntimeError("Missing gallery features. Run: python build_gallery_features.py")
    if not os.path.isdir(GALLERY_DIR):
        raise RuntimeError(f"Missing gallery folder: {GALLERY_DIR}")
    if not os.path.isdir(STATIC_DIR):
        raise RuntimeError(f"Missing static folder: {STATIC_DIR} (need static/index.html)")

    weights = np.load(WEIGHTS_PATH)
    MODEL = Dinov2Numpy(weights)

    feats = np.load(GALLERY_FEATURES_PATH).astype(np.float32)
    names = np.load(GALLERY_MAPPING_PATH)

    feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    GALLERY_NORM = feats_norm
    GALLERY_NAMES = names


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/history")
def api_history(limit: int = Query(50, ge=1, le=500)) -> Any:
    hist = _read_history()
    return JSONResponse({"ok": True, "items": hist[: int(limit)], "total": len(hist)})


@app.post("/api/history/clear")
def api_history_clear() -> Any:
    _atomic_write_json(HISTORY_PATH, [])
    return JSONResponse({"ok": True})


@app.post("/api/search")
async def api_search(
    image: UploadFile = File(...),
    topk: int = Query(DEFAULT_TOPK, ge=1, le=200),
    dedup: bool = Query(True),
) -> Any:
    global MODEL, GALLERY_NORM, GALLERY_NAMES
    assert MODEL is not None and GALLERY_NORM is not None and GALLERY_NAMES is not None

    suffix = os.path.splitext(image.filename or "")[1].lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"]:
        suffix = ".jpg"

    tmp_path = None
    t0 = time.time()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await image.read())

        # preprocess
        t_pre0 = time.time()
        pixel = resize_short_side(tmp_path)
        t_pre = (time.time() - t_pre0) * 1000

        # embed
        t_emb0 = time.time()
        q_feat = MODEL(pixel)
        q = l2_normalize(q_feat)
        t_emb = (time.time() - t_emb0) * 1000

        # search
        t_s0 = time.time()
        sims = GALLERY_NORM @ q
        N = int(sims.shape[0])

        raw_k = min(max(int(topk) * 6, int(topk) + 50), N)
        idx = np.argpartition(-sims, raw_k - 1)[:raw_k]
        idx = idx[np.argsort(-sims[idx])]
        t_search = (time.time() - t_s0) * 1000

        # dedup filter
        t_f0 = time.time()
        selected: List[dict] = []
        used_names = set()
        selected_vecs: List[np.ndarray] = []
        duplicates_removed = 0

        for i in idx:
            fname = str(GALLERY_NAMES[i])

            if dedup:
                if fname in used_names:
                    duplicates_removed += 1
                    continue

                v = GALLERY_NORM[i]
                if selected_vecs:
                    mat = np.vstack(selected_vecs)
                    if float(np.max(mat @ v)) >= DEDUP_FEAT_SIM_TH:
                        duplicates_removed += 1
                        continue

                used_names.add(fname)
                selected_vecs.append(v)

            selected.append({
                "score": float(sims[i]),
                "filename": fname,
                "url": f"/gallery_images/{fname}",
            })
            if len(selected) >= int(topk):
                break

        for r, it in enumerate(selected, 1):
            it["rank"] = r

        t_filter = (time.time() - t_f0) * 1000

        tiers = split_into_tiers_by_quantile(selected)

        total_ms = (time.time() - t0) * 1000

        payload = {
            "ok": True,
            "project": PROJECT_NAME,
            "topk_requested": int(topk),
            "topk_returned": int(len(selected)),
            "dedup_enabled": bool(dedup),
            "duplicates_removed": int(duplicates_removed),
            "tiers": tiers,
            "timing_ms": {
                "preprocess": round(t_pre, 2),
                "embed": round(t_emb, 2),
                "search": round(t_search, 2),
                "dedup_filter": round(t_filter, 2),
                "total": round(total_ms, 2),
            },
            "gallery_size": int(N),
        }

        # ---- Save history (best-effort) ----
        try:
            now = datetime.now()
            entry = {
                "id": uuid.uuid4().hex,
                "ts": now.isoformat(timespec="seconds"),
                "query_name": image.filename or "uploaded_image",
                "topk_requested": payload["topk_requested"],
                "topk_returned": payload["topk_returned"],
                "dedup_enabled": payload["dedup_enabled"],
                "duplicates_removed": payload["duplicates_removed"],
                "gallery_size": payload["gallery_size"],
                "timing_ms": payload["timing_ms"],
                "tiers": payload["tiers"],  # 直接存结果，前端可回放
            }
            _append_history(entry)
        except Exception:
            # 历史写入失败不影响检索主流程
            pass

        return JSONResponse(payload)

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
