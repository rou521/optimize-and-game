"""两个满足全局 L-smooth 假设的问题，以及手写反向传播的小型 MLP。"""
from dataclasses import dataclass
import numpy as np


@dataclass
class MatrixProblem:
    kind: str
    curvature: np.ndarray
    rotation: np.ndarray
    target: np.ndarray
    initial: np.ndarray
    alpha: float = 0.0

    @property
    def smoothness(self):
        # 二次项 Hessian 最大特征值 + 余弦项 Hessian 范数的上界。
        return float(self.curvature.max() + self.alpha)

    @property
    def lower_bound(self):
        # 两项都非负。非凸问题中这仅是下界，不宣称是最小值。
        return 0.0

    def value_grad(self, w):
        rotated = self.rotation @ (w - self.target)
        value = 0.5 * np.sum(self.curvature[:, None] * rotated**2)
        gradient = self.rotation.T @ (self.curvature[:, None] * rotated)
        if self.alpha:
            value += self.alpha * np.sum(1 - np.cos(w))
            gradient = gradient + self.alpha * np.sin(w)
        return float(value), gradient


def make_problem(kind="quadratic", seed=0, rows=12, cols=8):
    rng = np.random.default_rng(seed)
    if kind == "quadratic":
        curvature = np.geomspace(1, 30, rows)
        rotation, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
        alpha = 0.0
    elif kind == "nonconvex":
        # rotation=I 让非凸性可以直接核查：在 w_ij=pi 处，
        # 最小对角曲率 0.02-0.2=-0.18 < 0。
        curvature = np.geomspace(0.02, 1, rows)
        rotation = np.eye(rows)
        alpha = 0.2
    else:
        raise ValueError("未知问题类型。")
    target = rng.normal(size=(rows, cols)) / np.sqrt(rows)
    initial = rng.normal(size=(rows, cols))
    return MatrixProblem(kind, curvature, rotation, target, initial, alpha)


def mlp_initial(seed=0, inputs=10, hidden=24, outputs=3):
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(size=(inputs, hidden))/np.sqrt(inputs),
        "b1": np.zeros(hidden),
        "w2": rng.normal(size=(hidden, outputs))/np.sqrt(hidden),
        "b2": np.zeros(outputs),
    }


def mlp_value_grad(parameters, x, y):
    """f = mean((prediction-y)^2)/2；所有梯度与此归一化保持一致。"""
    hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
    residual = hidden @ parameters["w2"] + parameters["b2"] - y
    loss = float(np.mean(residual**2) / 2)
    d_output = residual / residual.size
    d_hidden = (d_output @ parameters["w2"].T) * (1-hidden**2)
    gradients = {
        "w2": hidden.T @ d_output,
        "b2": d_output.sum(axis=0),
        "w1": x.T @ d_hidden,
        "b1": d_hidden.sum(axis=0),
    }
    return loss, gradients


def mlp_data(seed=0, n_train=512, n_test=256):
    """固定 teacher 产生带噪声回归数据；测试集从不参与梯度计算。"""
    rng = np.random.default_rng(seed + 1000)
    teacher = mlp_initial(seed + 2000)
    x = rng.normal(size=(n_train+n_test, 10))
    hidden = np.tanh(x @ teacher["w1"] + teacher["b1"])
    y = hidden @ teacher["w2"] + teacher["b2"] + 0.03*rng.normal(size=(len(x), 3))
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]
