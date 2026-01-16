import numpy as np
import os
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

# 确保文件路径正确
weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

# 确保图片路径与你项目目录一致
cat_img = center_crop("./demo_data/cat.jpg")
dog_img = center_crop("./demo_data/dog.jpg")

cat_feat = vit(cat_img)
dog_feat = vit(dog_img)

ref_path = "./demo_data/cat_dog_feature.npy"
if os.path.exists(ref_path):
    ref_data = np.load(ref_path)
    # 验证第一张图 (猫) 的特征误差
    diff = np.abs(cat_feat[0] - ref_data[0]).max()
    print(f"特征误差是: {diff:.12f}")
