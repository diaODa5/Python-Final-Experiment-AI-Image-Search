import numpy as np
from scipy.ndimage import zoom


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.w, self.b, self.eps = weight, bias, eps

    def __call__(self, x):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return self.w * (x - mean) / np.sqrt(var + self.eps) + self.b


class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.scale = self.head_dim ** -0.5

        # 自动适配命名：有些权重包含 .attention.attention，有些是 .attention
        # 这里尝试匹配 query.weight
        q_key = f"{prefix}.query.weight"
        if q_key not in weights:
            # 尝试另一种常见的嵌套格式
            prefix = f"{prefix}.attention"

        self.qkv_w = np.concatenate([
            weights[f"{prefix}.query.weight"],
            weights[f"{prefix}.key.weight"],
            weights[f"{prefix}.value.weight"]
        ], axis=0)
        self.qkv_b = np.concatenate([
            weights[f"{prefix}.query.bias"],
            weights[f"{prefix}.key.bias"],
            weights[f"{prefix}.value.bias"]
        ], axis=0)

        # 投影层输出
        proj_prefix = prefix.replace("attention", "output.dense") if "output" not in prefix else prefix
        if f"{proj_prefix}.weight" not in weights:
            # 回退到 PPT 默认格式
            proj_prefix = f"{prefix.split('.attention')[0]}.attention.output.dense"

        # 终极兼容处理
        self.proj_w = weights.get(f"{proj_prefix}.weight", weights.get(f"{prefix}.proj.weight"))
        self.proj_b = weights.get(f"{proj_prefix}.bias", weights.get(f"{prefix}.proj.bias"))

    def __call__(self, x):
        B, N, D = x.shape
        qkv = (x @ self.qkv_w.T + self.qkv_b).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = softmax((q @ k.transpose(0, 1, 3, 2)) * self.scale)
        return (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D) @ self.proj_w.T + self.proj_b


class Dinov2Numpy:
    def __init__(self, weights):
        # 自动探测前缀 (有的权重以 dinov2. 开头)
        self.pre = "dinov2." if any("dinov2." in k for k in weights.keys()) else ""

        # 加载 Embeddings
        self.cls_token = weights[f"{self.pre}embeddings.cls_token"]
        self.pos_embed = weights[f"{self.pre}embeddings.position_embeddings"]
        self.patch_w = weights[f"{self.pre}embeddings.patch_embeddings.projection.weight"]
        self.patch_b = weights[f"{self.pre}embeddings.patch_embeddings.projection.bias"]

        h_size = self.cls_token.shape[-1]
        self.config = {"hidden_size": h_size, "num_heads": h_size // 64}

        # 加载 Blocks
        self.blocks = []
        i = 0
        while True:
            p = f"{self.pre}encoder.layer.{i}"
            if f"{p}.norm1.weight" not in weights:
                break

            # 适配 LayerScale 名：lambda1 或 gamma
            ls1 = weights.get(f"{p}.layer_scale1.lambda1", weights.get(f"{p}.ls1.gamma", 1.0))
            ls2 = weights.get(f"{p}.layer_scale2.lambda1", weights.get(f"{p}.ls2.gamma", 1.0))

            self.blocks.append({
                "n1": LayerNorm(weights[f"{p}.norm1.weight"], weights[f"{p}.norm1.bias"]),
                "at": MultiHeadAttention(self.config, f"{p}.attention", weights),
                "ls1": ls1,
                "n2": LayerNorm(weights[f"{p}.norm2.weight"], weights[f"{p}.norm2.bias"]),
                "mlp_w1": weights[f"{p}.mlp.fc1.weight"], "mlp_b1": weights[f"{p}.mlp.fc1.bias"],
                "mlp_w2": weights[f"{p}.mlp.fc2.weight"], "mlp_b2": weights[f"{p}.mlp.fc2.bias"],
                "ls2": ls2
            })
            i += 1

        self.norm = LayerNorm(weights[f"{self.pre}layernorm.weight"], weights[f"{self.pre}layernorm.bias"])

    def __call__(self, x):
        B, C, H, W = x.shape
        p = 14

        # Patch Embed
        # 这里假设 H/W 已经是 patch_size 的整数倍（你的 preprocess 会保证）
        gh = H // p
        gw = W // p

        x = x.reshape(B, C, gh, p, gw, p).transpose(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        x = x @ self.patch_w.reshape(self.config['hidden_size'], -1).T + self.patch_b

        # 位置编码插值：支持“非正方形”输入（例如 224x336 等）
        # 原实现用 sqrt(token_num) 当作 new_grid，会在长方形输入时导致广播错误。
        if x.shape[1] + 1 != self.pos_embed.shape[1]:
            cls_pos = self.pos_embed[:, :1, :].astype(np.float32)
            patch_pos = self.pos_embed[:, 1:, :].astype(np.float32)  # (1, oldN, D)

            old_grid = int(np.sqrt(patch_pos.shape[1]))
            if old_grid * old_grid != patch_pos.shape[1]:
                old_grid = int(round(np.sqrt(patch_pos.shape[1])))
                if old_grid * old_grid != patch_pos.shape[1]:
                    raise ValueError(
                        f"Position embedding patch tokens not square: {patch_pos.shape[1]}"
                    )

            # (1, oldN, D) -> (1, D, old_grid, old_grid)
            patch_pos_2d = patch_pos.reshape(1, old_grid, old_grid, -1).transpose(0, 3, 1, 2)

            # zoom 到 (gh, gw)（注意这里是长方形）
            patch_pos_2d = zoom(
                patch_pos_2d,
                (1, 1, gh / old_grid, gw / old_grid),
                order=1
            )

            # zoom 可能因为浮点比例产生 +-1 的尺寸偏差，强制裁剪到精确形状
            patch_pos_2d = patch_pos_2d[:, :, :gh, :gw]

            # (1, D, gh, gw) -> (1, gh*gw, D)
            patch_pos_new = patch_pos_2d.transpose(0, 2, 3, 1).reshape(1, gh * gw, -1)

            pos_embed = np.concatenate([cls_pos, patch_pos_new], axis=1).astype(np.float32)
        else:
            pos_embed = self.pos_embed.astype(np.float32)

        x = np.concatenate([np.broadcast_to(self.cls_token, (B, 1, x.shape[-1])), x], axis=1).astype(np.float32) + pos_embed

        for b in self.blocks:
            x = x + b["ls1"] * b["at"](b["n1"](x))
            res = b["n2"](x)
            mlp = gelu(res @ b["mlp_w1"].T + b["mlp_b1"]) @ b["mlp_w2"].T + b["mlp_b2"]
            x = x + b["ls2"] * mlp

        return self.norm(x)[:, 0]
