import os
import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

def l2_normalize(x, eps=1e-12):
    x = x.reshape(-1).astype(np.float32)
    return x / (np.linalg.norm(x) + eps)

def search_topk(
    query_img_path: str,
    topk: int = 50,
    weights_path: str = "vit-dinov2-base.npz",
    gallery_feat_path: str = "gallery_features.npy",
    gallery_map_path: str = "gallery_mapping.npy",
    gallery_img_dir: str = "gallery_images",
    save_txt: str | None = "top50_results.txt",
):
    # 1) load model
    weights = np.load(weights_path)
    model = Dinov2Numpy(weights)

    # 2) load gallery features + mapping
    gallery_feats = np.load(gallery_feat_path).astype(np.float32)   # (N,768)
    gallery_names = np.load(gallery_map_path)                      # (N,)

    # 3) preprocess query (必须和建库一致：短边224 + 14倍数约束)
    q_pixel = resize_short_side(query_img_path)  # preprocess_image.py 已按 224 & 14 倍数实现 :contentReference[oaicite:2]{index=2}
    q_feat = model(q_pixel)                      # (1,768)
    q = l2_normalize(q_feat)

    # 4) cosine similarity: normalize then dot
    g = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-12)
    sims = g @ q  # (N,)

    # 5) Top-K
    k = min(int(topk), sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    results = []
    for i in idx:
        fname = str(gallery_names[i])
        img_path = os.path.join(gallery_img_dir, fname)
        results.append((img_path, float(sims[i])))

    # 6) optional save
    if save_txt:
        with open(save_txt, "w", encoding="utf-8") as f:
            for p, s in results:
                f.write(f"{s:.6f}\t{p}\n")

    return results

if __name__ == "__main__":
    res = search_topk("./demo_data/cat.jpg", topk=50)
    for p, s in res:
        print(f"{s:.4f}\t{p}")
