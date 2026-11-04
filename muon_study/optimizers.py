"""用 NumPy 展开 Muon 的每一步，便于逐行对应报告公式。

这是 CPU/float64 教学实现。没有模拟官方 GPU/bfloat16 的运行速度。
svd 路径用于理论验证；ns 路径采用作者公开实现的五阶多项式。
"""
from dataclasses import dataclass
import numpy as np


def _matrix(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError("Muon 需要非空二维矩阵；偏置和向量请使用其他优化器。")
    if not np.isfinite(value).all():
        raise ValueError("输入含 NaN 或 Inf。")
    return value


def polar_svd(matrix):
    """紧致 SVD 的极分解因子 U_r V_r^T，零矩阵映射为零。

    为与紧致极分解定义一致，不返回秩亏矩阵的完整 U @ Vt；
    完整乘积会额外补上零奇异值方向，此处选择不更新这些方向。
    实数理论保留所有正奇异值；浮点计算用通常的数值秩阈值。
    """
    matrix = _matrix(matrix)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular[0]
    keep = singular > tolerance
    return u[:, keep] @ vt[keep, :]


def newton_schulz(matrix, steps=5, eps=1e-7):
    """Muon 作者使用的 quintic Newton–Schulz 更新（双精度版本）。

    先除以 Frobenius 范数，保证初始谱范数不超过 1；高矩阵先转置，
    从而让 X @ X.T 在较小维度计算。注意：这组系数不是逐步趋于
    精确极分解的通用求解器，多迭代也不保证误差单调下降。
    """
    matrix = _matrix(matrix)
    if not isinstance(steps, int) or steps < 0 or eps <= 0:
        raise ValueError("steps 必须为非负整数，eps 必须为正数。")
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.T.copy() if transposed else matrix.copy()
    x /= np.linalg.norm(x, "fro") + eps
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


@dataclass
class Muon:
    """单个二维参数的 Muon；每个参数必须有独立的实例/动量。

    M_t = beta M_{t-1} + (1-beta) G_t
    H_t = (1-beta) G_t + beta M_t  （nesterov=True）
    D_t = polar(H_t) 或 NS(H_t)
    W_{t+1} = (1-lr*weight_decay) W_t - lr*scale*D_t

    理论实验：scale_mode='none', weight_decay=0, method='svd'。
    作者实现的形状缩放：scale_mode='jordan'。
    Moonlight 报告的 RMS 缩放：scale_mode='moonlight'。
    两种缩放是不同约定，不应同时使用。
    """
    beta: float = 0.95
    method: str = "ns"
    ns_steps: int = 5
    nesterov: bool = True
    weight_decay: float = 0.0
    scale_mode: str = "none"

    def __post_init__(self):
        if not 0 <= self.beta < 1:
            raise ValueError("beta 必须在 [0,1) 内。")
        if self.method not in ("svd", "ns"):
            raise ValueError("method 必须为 svd 或 ns。")
        if self.scale_mode not in ("none", "jordan", "moonlight"):
            raise ValueError("未知 scale_mode。")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 不得为负。")
        self.momentum = None
        self.last_input = None

    def direction(self, gradient):
        gradient = _matrix(gradient)
        if self.momentum is None:
            self.momentum = np.zeros_like(gradient)
        if gradient.shape != self.momentum.shape:
            raise ValueError("同一个 Muon 实例不能用于不同形状的参数。")
        self.momentum = self.beta * self.momentum + (1 - self.beta) * gradient
        h = ((1 - self.beta) * gradient + self.beta * self.momentum
             if self.nesterov else self.momentum.copy())
        self.last_input = h.copy()  # 留存 H_t，方便记录动量误差和正交化误差。
        d = polar_svd(h) if self.method == "svd" else newton_schulz(h, self.ns_steps)
        rows, cols = gradient.shape
        if self.scale_mode == "jordan":
            d *= max(1.0, rows / cols) ** 0.5
        elif self.scale_mode == "moonlight":
            d *= 0.2 * max(rows, cols) ** 0.5
        return d

    def step(self, parameter, gradient, lr):
        """返回新参数，不原地覆盖调用方的参数或梯度。"""
        parameter, gradient = _matrix(parameter), _matrix(gradient)
        if parameter.shape != gradient.shape or not np.isfinite(lr) or lr < 0:
            raise ValueError("参数/梯度形状须一致，学习率须有限且非负。")
        return (1 - lr * self.weight_decay) * parameter - lr * self.direction(gradient)


class AdamW:
    """用于 MLP 输出层和偏置的辅助优化器，也提供完整 AdamW 对照。"""
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self, parameters, gradients, lr):
        self.t += 1
        result = {}
        for name, p in parameters.items():
            g = gradients[name]
            self.m[name] = self.beta1 * self.m.get(name, np.zeros_like(p)) + (1-self.beta1)*g
            self.v[name] = self.beta2 * self.v.get(name, np.zeros_like(p)) + (1-self.beta2)*g*g
            mhat = self.m[name] / (1-self.beta1**self.t)
            vhat = self.v[name] / (1-self.beta2**self.t)
            result[name] = (1-lr*self.weight_decay)*p - lr*mhat/(np.sqrt(vhat)+self.eps)
        return result
