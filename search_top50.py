#!/usr/bin/env python3
import os
import argparse
import shutil
import numpy as np

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side


def _l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.reshape(-1).astype(np.float32)
    return v / (np.linalg.norm(v) + eps)


def search_topk(
    query_img_path: str,
    topk: int = 50,
    weights_path: str = "vit-dinov2-base.npz",
    gallery_feat_path: str = "gallery_features.npy",
    gallery_map_path: str = "gallery_mapping.npy",
    gallery_img_dir: str = "gallery_images",
):
    # load model
    weights = np.load(weights_path)
    model = Dinov2Numpy(weights)

    # load gallery
    gallery_feats = np.load(gallery_feat_path).astype(np.float32)   # (N,768)
    gallery_names = np.load(gallery_map_path)                       # (N,)

    # preprocess + extract query
    q_pixel = resize_short_side(query_img_path)
    q_feat = model(q_pixel)  # (1,768) or (768,)
    q = _l2norm(q_feat)

    # cosine similarity via normalized dot product
    g = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-12)
    sims = g @ q  # (N,)

    k = min(int(topk), sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    results = []
    for rank, i in enumerate(idx, 1):
        fname = str(gallery_names[i])
        img_path = os.path.join(gallery_img_dir, fname)
        results.append((rank, float(sims[i]), img_path))
    return results


def main():
    ap = argparse.ArgumentParser(description="Image-to-image retrieval (Top-K) using DINOv2 numpy features.")
    ap.add_argument("--query", required=True, help="Path to the query image.")
    ap.add_argument("--topk", type=int, default=50, help="Return Top-K results (default: 50).")
    ap.add_argument("--weights", default="vit-dinov2-base.npz", help="Weights npz path.")
    ap.add_argument("--gallery_features", default="gallery_features.npy", help="Gallery feature matrix .npy")
    ap.add_argument("--gallery_mapping", default="gallery_mapping.npy", help="Gallery filename mapping .npy")
    ap.add_argument("--gallery_dir", default="gallery_images", help="Gallery images folder.")
    ap.add_argument("--save_txt", default="top50_results.txt", help="Save ranked results to a txt file.")
    ap.add_argument("--export_dir", default="", help="If set, copy Top-K images into this folder for easy viewing.")
    args = ap.parse_args()

    results = search_topk(
        query_img_path=args.query,
        topk=args.topk,
        weights_path=args.weights,
        gallery_feat_path=args.gallery_features,
        gallery_map_path=args.gallery_mapping,
        gallery_img_dir=args.gallery_dir,
    )

    # print + save
    if args.save_txt:
        with open(args.save_txt, "w", encoding="utf-8") as f:
            for rank, score, path in results:
                line = f"{rank:02d}\t{score:.6f}\t{path}"
                print(line)
                f.write(line + "\n")
    else:
        for rank, score, path in results:
            print(f"{rank:02d}\t{score:.6f}\t{path}")

    # optional export images
    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)
        for rank, score, path in results:
            ext = os.path.splitext(path)[1] or ".jpg"
            dst = os.path.join(args.export_dir, f"{rank:02d}_{score:.4f}{ext}")
            if os.path.exists(path):
                shutil.copy2(path, dst)
        print(f"\nExported Top-{len(results)} images to: {args.export_dir}")


if __name__ == "__main__":
    main()
