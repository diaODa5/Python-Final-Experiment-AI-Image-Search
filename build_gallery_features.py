import os
import numpy as np
from tqdm import tqdm
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

# --- 配置 ---
IMG_DIR = "gallery_images"
WEIGHTS_PATH = "vit-dinov2-base.npz"
SAVE_PATH = "gallery_features.npy"


def build():
    weights = np.load(WEIGHTS_PATH)
    model = Dinov2Numpy(weights)

    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    all_features = []
    file_mapping = []

    print("开始提取 Gallery 特征...")
    for filename in tqdm(image_files):
        img_path = os.path.join(IMG_DIR, filename)
        try:
            # 使用 resize_short_side 处理变长图片
            pixel_values = resize_short_side(img_path)
            feat = model(pixel_values)  # 得到 (1, 768)
            all_features.append(feat)
            file_mapping.append(filename)
        except Exception as e:
            print(f"处理 {filename} 出错: {e}")

    # 保存特征矩阵 (N, 768)
    np.save(SAVE_PATH, np.vstack(all_features))
    # 保存文件名映射，方便检索后找到原图
    np.save("gallery_mapping.npy", np.array(file_mapping))
    print(f"特征库已构建完成，保存至 {SAVE_PATH}")


if __name__ == "__main__":
    build()