"""运行全部实验并保存原始 CSV、汇总 JSON、PNG/PDF 图及 LaTeX 数值。

用法：python -m muon_study.experiments --output results
每个 horizon 都从同一初值重新运行，并使用对应的 eta(T)。
不能把一条固定学习率轨迹的不同前缀当作 O(T^-1/2) 定理的验证。
"""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/muon-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .optimizers import AdamW, Muon, newton_schulz, polar_svd
from .problems import make_problem, mlp_data, mlp_initial, mlp_value_grad

COLORS = ["#1769aa", "#dc6b27", "#14806b", "#8654aa", "#555555"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140,
                     "axes.prop_cycle": matplotlib.cycler(color=COLORS)})


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def convergence_run(problem, horizon, beta, nesterov=True):
    """验证精确极分解的动量定理；返回可独立重算界的所有常数。"""
    w = problem.initial.copy()
    initial_value, initial_gradient = problem.value_grad(w)
    delta = initial_value - problem.lower_bound
    rank_bound = min(w.shape)
    smoothness = problem.smoothness
    q = beta if nesterov else 1.0
    gamma = 0.5 + 2*q*beta/(1-beta)
    eta = np.sqrt(delta/(smoothness*rank_bound*gamma*horizon))
    optimizer = Muon(beta=beta, method="svd", nesterov=nesterov)
    nuclear_sum = 0.0
    worst_descent_residual = -np.inf
    for _ in range(horizon):
        value, gradient = problem.value_grad(w)
        nuclear = float(np.linalg.svd(gradient, compute_uv=False).sum())
        d = optimizer.direction(gradient)
        error = np.linalg.norm(optimizer.last_input-gradient, "fro")
        new_w = w - eta*d
        new_value, _ = problem.value_grad(new_w)
        # 残差 <= 0 表示这一步符合下降引理 + 动量误差界。
        residual = new_value-value+eta*nuclear-2*eta*np.sqrt(rank_bound)*error
        residual -= smoothness*rank_bound*eta**2/2
        worst_descent_residual = max(worst_descent_residual, residual)
        nuclear_sum += nuclear
        w = new_w
    transient = 2*np.sqrt(rank_bound)*q*beta*np.linalg.norm(initial_gradient, "fro")
    transient /= horizon*(1-beta)
    bound = delta/(eta*horizon) + smoothness*rank_bound*gamma*eta + transient
    measured = nuclear_sum/horizon
    tolerance = 1e-9 * max(1.0, initial_value, bound)
    if measured > bound+tolerance or worst_descent_residual > tolerance:
        raise AssertionError("收敛界验证失败，请先排查算法/常数/索引。")
    return dict(horizon=horizon, beta=beta, nesterov=nesterov, eta=eta,
                L=smoothness, r=rank_bound, delta=delta, gamma=gamma,
                transient=transient, mean_nuclear_gradient=measured,
                theoretical_bound=bound, ratio=measured/bound,
                max_descent_residual=worst_descent_residual,
                final_value=problem.value_grad(w)[0])


def rate_experiment(output, seeds, horizons):
    rows = []
    for kind in ("quadratic", "nonconvex"):
        for beta in (0.0, 0.9):
            for seed in seeds:
                problem = make_problem(kind, seed)
                for horizon in horizons:
                    rows.append(dict(problem=kind, seed=seed,
                                     **convergence_run(problem, horizon, beta)))
            print(f"rate: {kind}, beta={beta} complete", flush=True)
    write_csv(output/"rates.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    slopes = {}
    for ax, kind in zip(axes, ("quadratic", "nonconvex")):
        for color, beta in zip(COLORS, (0.0, 0.9)):
            data = [r for r in rows if r["problem"] == kind and r["beta"] == beta]
            measured = np.array([[r["mean_nuclear_gradient"] for r in data
                                  if r["horizon"] == t] for t in horizons])
            bounds = np.array([[r["theoretical_bound"] for r in data
                                if r["horizon"] == t] for t in horizons])
            slope = float(np.polyfit(np.log(horizons), np.log(measured.mean(axis=1)), 1)[0])
            slopes[f"{kind}_beta_{beta}"] = slope
            ax.loglog(horizons, measured.mean(axis=1), "o-", color=color,
                      label=f"exact, beta={beta:g}; fit {slope:.2f}")
            ax.fill_between(horizons, measured.min(axis=1), measured.max(axis=1),
                            color=color, alpha=0.12)
            ax.loglog(horizons, bounds.mean(axis=1), "--", color=color,
                      label=f"bound, beta={beta:g}")
        ax.set(title=kind.capitalize(), xlabel="Independent run length T",
               ylabel="Mean nuclear gradient norm")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    save_figure(fig, output, "01_convergence_rate")
    return dict(slopes=slopes, runs=len(rows), max_ratio=max(r["ratio"] for r in rows),
                bound_violations=int(sum(r["ratio"] > 1+1e-9 for r in rows)),
                max_descent_residual=max(r["max_descent_residual"] for r in rows))


def trajectory_experiment(output, seeds, horizon=2048):
    """定步长平台与衰减步长；同时核查 NS 的含误差项事后界。"""
    variants = {
        "SVD, constant": ("svd", 0.0, False),
        "SVD, decaying": ("svd", 0.0, True),
        "NS5+momentum, constant": ("ns", 0.9, False),
        "NS5+momentum, decaying": ("ns", 0.9, True),
    }
    rows, checks = [], []
    for seed in seeds:
        problem = make_problem("quadratic", seed)
        r, L = min(problem.initial.shape), problem.smoothness
        initial_value = problem.value_grad(problem.initial)[0]
        # 同一个问题所有变体共享 eta0，不根据测试结果选择赢家。
        eta0 = np.sqrt(2*initial_value/(L*r*horizon))
        for label, (method, beta, decaying) in variants.items():
            optimizer = Muon(beta=beta, method=method)
            w = problem.initial.copy()
            lhs = 0.0
            errors = 0.0
            eta_sum = 0.0
            delta_sum = 0.0
            max_residual = -np.inf
            for t in range(horizon):
                value, gradient = problem.value_grad(w)
                norm_star = np.linalg.svd(gradient, compute_uv=False).sum()
                norm_f = np.linalg.norm(gradient, "fro")
                eta = eta0/(1+t/64)**0.6 if decaying else eta0
                d = optimizer.direction(gradient)
                h = optimizer.last_input
                momentum_error = np.linalg.norm(h-gradient, "fro")
                polar_error = np.linalg.norm(d-polar_svd(h), "fro")
                penalty = 2*np.sqrt(r)*momentum_error + norm_f*polar_error
                penalty += L*eta*np.linalg.norm(d, "fro")**2/2
                new_w = w-eta*d
                new_value = problem.value_grad(new_w)[0]
                max_residual = max(max_residual, new_value-value+eta*(norm_star-penalty))
                lhs += eta*norm_star
                errors += eta*penalty
                eta_sum += eta
                delta_sum += polar_error
                if t % 8 == 0 or t == horizon-1:
                    rows.append(dict(seed=seed, variant=label, step=t, eta=eta,
                                     value=value, gradient_fro=norm_f,
                                     gradient_nuclear=norm_star, polar_error=polar_error))
                w = new_w
            rhs = initial_value + errors  # f_lower = 0；变步长使用加权平均。
            if lhs > rhs+1e-8*max(1, rhs) or max_residual > 1e-8*max(1, initial_value):
                raise AssertionError("含近似误差的下降界验证失败。")
            checks.append(dict(seed=seed, variant=label, weighted_mean=lhs/eta_sum,
                               a_posteriori_bound=rhs/eta_sum, ratio=lhs/rhs,
                               mean_polar_error=delta_sum/horizon,
                               final_value=problem.value_grad(w)[0],
                               max_descent_residual=max_residual))
    write_csv(output/"trajectories.csv", rows)
    write_csv(output/"inexact_checks.csv", checks)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for label, color in zip(variants, COLORS):
        data = [r for r in rows if r["variant"] == label]
        steps = sorted(set(r["step"] for r in data))
        for ax, key in zip(axes, ("value", "gradient_fro")):
            values = np.array([[r[key] for r in data if r["step"] == t] for t in steps])
            ax.semilogy(steps, values.mean(axis=1), color=color, label=label)
            ax.fill_between(steps, values.min(axis=1), values.max(axis=1), color=color, alpha=.1)
    axes[0].set(xlabel="Update step", ylabel="Objective gap (known optimum = 0)", title="Quadratic objective")
    axes[1].set(xlabel="Update step", ylabel="Frobenius gradient norm", title="Fixed-step plateau vs decay")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save_figure(fig, output, "02_optimization_trajectories")
    return {label: dict(mean_final_value=float(np.mean([r["final_value"] for r in checks if r["variant"] == label])),
                        mean_polar_error=float(np.mean([r["mean_polar_error"] for r in checks if r["variant"] == label])),
                        max_bound_ratio=max(r["ratio"] for r in checks if r["variant"] == label))
            for label in variants}


def spectrum_experiment(output):
    sigma = np.geomspace(1e-4, 1, 48)
    matrix = np.diag(sigma)
    exact = polar_svd(matrix)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    errors = {}
    for iterations, color in zip((1, 2, 5, 10, 20), COLORS):
        d = newton_schulz(matrix, iterations)
        errors[str(iterations)] = float(np.linalg.norm(d-exact, "fro")/np.linalg.norm(exact, "fro"))
        axes[0].semilogx(sigma, np.diag(d), color=color, label=f"NS {iterations} steps")
        for s, transformed in zip(sigma, np.diag(d)):
            rows.append(dict(ns_steps=iterations, input_singular=s, output_coefficient=transformed,
                             relative_polar_error=errors[str(iterations)]))
    axes[0].axhline(1, color="black", linestyle="--", label="Exact polar")
    axes[0].set(xlabel="Input singular value (before normalization)",
                ylabel="Coefficient in the original singular direction", title="NS is a spectral transformation")
    axes[0].legend(fontsize=8)
    # 一维反例：即使是强凸二次函数，定步长符号更新也可能往复振荡。
    for decaying, color in zip((False, True), COLORS):
        x, history = 0.35, []
        for t in range(80):
            history.append(x)
            eta = 0.2/(t+1)**0.6 if decaying else 0.2
            x -= eta*np.sign(x)
        axes[1].plot(history, color=color, label="Decaying step" if decaying else "Constant step 0.2")
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(xlabel="Update step", ylabel="x", title="Sanity check: f(x) = x²/2")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    write_csv(output/"ns_spectrum.csv", rows)
    save_figure(fig, output, "03_ns_and_step_size")
    return errors


def neural_experiment(output, seeds, steps=800):
    rows, final = [], []
    for seed in seeds:
        x, y, xt, yt = mlp_data(seed)
        # 两种优化器使用完全相同的初值、训练样本和每步 mini-batch。
        batches = np.random.default_rng(seed+3000).integers(0, len(x), size=(steps, 64))
        for method in ("Muon NS5 + auxiliary AdamW", "AdamW"):
            parameters = mlp_initial(seed)
            muon = Muon(beta=.95, method="ns", scale_mode="jordan", weight_decay=.01)
            auxiliary = AdamW(weight_decay=.01)
            for t in range(steps+1):
                if t % 20 == 0:
                    train_loss = mlp_value_grad(parameters, x, y)[0]
                    test_loss = mlp_value_grad(parameters, xt, yt)[0]
                    rows.append(dict(seed=seed, method=method, step=t,
                                     train_loss=train_loss, test_loss=test_loss))
                if t == steps:
                    final.append(dict(seed=seed, method=method, train_loss=train_loss, test_loss=test_loss))
                    break
                _, gradients = mlp_value_grad(parameters, x[batches[t]], y[batches[t]])
                # 固定预设学习率；不使用测试集调参，不宣称两者已充分调优。
                if method.startswith("Muon"):
                    hidden = muon.step(parameters["w1"], gradients["w1"], lr=.02)
                    aux_params = {k: v for k, v in parameters.items() if k != "w1"}
                    parameters = auxiliary.step(aux_params, gradients, lr=.003)
                    parameters["w1"] = hidden
                else:
                    parameters = auxiliary.step(parameters, gradients, lr=.003)
            print(f"MLP: seed={seed}, {method} complete", flush=True)
    write_csv(output/"mlp.csv", rows)
    write_csv(output/"mlp_final.csv", final)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for method, color in zip(("Muon NS5 + auxiliary AdamW", "AdamW"), COLORS):
        data = [r for r in rows if r["method"] == method]
        ticks = sorted(set(r["step"] for r in data))
        for ax, key in zip(axes, ("train_loss", "test_loss")):
            values = np.array([[r[key] for r in data if r["step"] == t] for t in ticks])
            ax.semilogy(ticks, values.mean(axis=1), color=color, label=method)
            ax.fill_between(ticks, values.min(axis=1), values.max(axis=1), color=color, alpha=.12)
    for ax, title in zip(axes, ("Training loss", "Held-out test loss")):
        ax.set(title=title, xlabel="Mini-batch update step", ylabel="Half mean squared error")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    save_figure(fig, output, "04_neural_network")
    return {method: {key: float(np.mean([r[key] for r in final if r["method"] == method]))
                     for key in ("train_loss", "test_loss")}
            for method in ("Muon NS5 + auxiliary AdamW", "AdamW")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, default=5, help="随机种子数，使用 0,...,N-1")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("seeds 必须至少为 1")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seeds, horizons = list(range(args.seeds)), [64, 128, 256, 512, 1024, 2048, 4096]
    summary = {"seeds": seeds, "horizons": horizons,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "matplotlib": matplotlib.__version__},
               "rate": rate_experiment(output, seeds, horizons),
               "trajectories": trajectory_experiment(output, seeds),
               "ns_relative_error": spectrum_experiment(output),
               "mlp": neural_experiment(output, seeds)}
    summary["elapsed_seconds"] = time.perf_counter()-start
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    # 报告读取真实运行值，不手工复制或构造数字。
    macros = {
        "NumSeeds": args.seeds, "NumRuns": summary["rate"]["runs"],
        "MaxRatio": f'{summary["rate"]["max_ratio"]:.4f}',
        "NumViolations": summary["rate"]["bound_violations"],
        "SlopeQuadratic": f'{summary["rate"]["slopes"]["quadratic_beta_0.0"]:.3f}',
        "SlopeNonconvex": f'{summary["rate"]["slopes"]["nonconvex_beta_0.0"]:.3f}',
        "SlopeQuadraticMomentum": f'{summary["rate"]["slopes"]["quadratic_beta_0.9"]:.3f}',
        "SlopeNonconvexMomentum": f'{summary["rate"]["slopes"]["nonconvex_beta_0.9"]:.3f}',
        "NeuralMuonTest": f'{summary["mlp"]["Muon NS5 + auxiliary AdamW"]["test_loss"]:.6f}',
        "NeuralAdamTest": f'{summary["mlp"]["AdamW"]["test_loss"]:.6f}',
        "NSFiveError": f'{summary["ns_relative_error"]["5"]:.3f}',
    }
    (output/"metrics.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in macros.items())+"\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
