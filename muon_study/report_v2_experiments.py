"""报告 v2：同初值、同学习率的 2×2×2 因子对照，不覆盖原始结果。

python -m muon_study.report_v2_experiments --output results_v2
速率图仅重绘 results/rates.csv；控制实验重新计算全部轨迹。
"""
import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np

from .experiments import COLORS, plt, save_figure, write_csv
import matplotlib
from .optimizers import Muon, polar_svd
from .problems import make_problem


def budget_plot(source, output):
    with (source / "rates.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for col, kind in enumerate(("quadratic", "nonconvex")):
        for beta, color in zip((0.0, 0.9), COLORS):
            data = [r for r in rows if r["problem"] == kind and float(r["beta"]) == beta]
            budgets = sorted({int(r["horizon"]) for r in data})
            for row, key in enumerate(("mean_nuclear_gradient", "eta")):
                values = np.array([[float(r[key]) for r in data if int(r["horizon"]) == t]
                                   for t in budgets])
                ax = axes[row, col]
                ax.loglog(budgets, values.mean(axis=1), "o-", color=color,
                          label=f"SVD, beta={beta:g}")
                ax.fill_between(budgets, values.min(axis=1), values.max(axis=1),
                                color=color, alpha=.12)
            bounds = np.array([[float(r["theoretical_bound"]) for r in data
                                if int(r["horizon"]) == t] for t in budgets])
            axes[0, col].loglog(budgets, bounds.mean(axis=1), "--", color=color,
                               label=f"Bound, beta={beta:g}")
        axes[0, col].set(title=kind.capitalize(), ylabel="Time-mean nuclear gradient norm")
        axes[1, col].set(ylabel="Learning rate eta(T)")
    for ax in axes.flat:
        ax.set_xlabel("Independent budget T (new run)")
        ax.grid(True, which="both", alpha=.2)
        ax.legend(fontsize=10)
    save_figure(fig, output, "01_budget_and_learning_rate")


def controlled_experiment(output, seeds, horizon):
    rows, final = [], []
    for kind in ("quadratic", "nonconvex"):
        for seed in seeds:
            problem = make_problem(kind, seed)
            initial_value = problem.value_grad(problem.initial)[0]
            delta = initial_value - problem.lower_bound
            r, L = min(problem.initial.shape), problem.smoothness
            eta0 = np.sqrt(2 * delta / (L * r * horizon))
            for schedule in ("constant", "decaying"):
                for method in ("svd", "ns"):
                    for beta in (0.0, 0.9):
                        opt = Muon(beta=beta, method=method, ns_steps=5,
                                   nesterov=True, scale_mode="none", weight_decay=0)
                        w = problem.initial.copy()
                        weighted_sum = error_sum = eta_sum = polar_sum = 0.0
                        worst_residual = -np.inf
                        for t in range(horizon + 1):
                            value, g = problem.value_grad(w)
                            norm_f = float(np.linalg.norm(g, "fro"))
                            norm_star = float(np.linalg.svd(g, compute_uv=False).sum())
                            # 明确记录 W_T；该行不是额外的一次参数更新。
                            if t == horizon:
                                rows.append(dict(problem=kind, seed=seed, schedule=schedule,
                                                 method=method, beta=beta, step=t,
                                                 value=value, gradient_fro=norm_f,
                                                 gradient_nuclear=norm_star, eta="",
                                                 momentum_error="", polar_error="",
                                                 direction_fro="", inner_product=""))
                                break
                            eta = eta0 if schedule == "constant" else eta0 / (1 + t / 64) ** .6
                            d = opt.direction(g)
                            momentum_error = float(np.linalg.norm(opt.last_input - g, "fro"))
                            polar_error = float(np.linalg.norm(d - polar_svd(opt.last_input), "fro"))
                            norm_d = float(np.linalg.norm(d, "fro"))
                            inner = float(np.sum(g * d))
                            penalty = 2 * np.sqrt(r) * momentum_error + norm_f * polar_error
                            penalty += L * eta * norm_d ** 2 / 2
                            new_w = w - eta * d
                            new_value = problem.value_grad(new_w)[0]
                            residual = new_value - value + eta * (norm_star - penalty)
                            worst_residual = max(worst_residual, residual)
                            weighted_sum += eta * norm_star
                            error_sum += eta * penalty
                            eta_sum += eta
                            polar_sum += polar_error
                            if t % 8 == 0:
                                rows.append(dict(problem=kind, seed=seed, schedule=schedule,
                                                 method=method, beta=beta, step=t,
                                                 value=value, gradient_fro=norm_f,
                                                 gradient_nuclear=norm_star, eta=eta,
                                                 momentum_error=momentum_error,
                                                 polar_error=polar_error, direction_fro=norm_d,
                                                 inner_product=inner))
                            w = new_w
                        bound = (delta + error_sum) / eta_sum
                        measured = weighted_sum / eta_sum
                        tolerance = 1e-8 * max(1.0, initial_value, bound)
                        if measured > bound + tolerance or worst_residual > tolerance:
                            raise AssertionError("v2 控制实验违反含误差下降界")
                        final.append(dict(problem=kind, seed=seed, schedule=schedule,
                                          method=method, beta=beta, horizon=horizon,
                                          eta0=eta0, final_value=value, final_gradient_fro=norm_f,
                                          weighted_mean_nuclear=measured, posterior_bound=bound,
                                          bound_ratio=measured / bound,
                                          mean_polar_error=polar_sum / horizon,
                                          max_descent_residual=worst_residual))
        print(f"controlled: {kind} complete", flush=True)
    # 检查每一对照组确实使用同一 eta0，且所有最终行均对应 W_T。
    for kind in ("quadratic", "nonconvex"):
        for seed in seeds:
            group = [r for r in final if r["problem"] == kind and r["seed"] == seed]
            assert len(group) == 8 and len({r["eta0"] for r in group}) == 1
    write_csv(output / "controlled_trajectories.csv", rows)
    write_csv(output / "controlled_final.csv", final)
    summary = []
    for kind in ("quadratic", "nonconvex"):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
        for i, schedule in enumerate(("constant", "decaying")):
            for (method, beta), color in zip((("svd", 0.), ("svd", .9), ("ns", 0.), ("ns", .9)), COLORS):
                group = [r for r in rows if r["problem"] == kind and r["schedule"] == schedule
                         and r["method"] == method and r["beta"] == beta]
                ticks = sorted({r["step"] for r in group})
                label = f'{"SVD" if method == "svd" else "NS5"}, beta={beta:g}'
                for j, key in enumerate(("value", "gradient_fro")):
                    values = np.array([[r[key] for r in group if r["step"] == t] for t in ticks])
                    ax = axes[i, j]
                    ax.semilogy(ticks, values.mean(axis=1), color=color, label=label)
                    ax.fill_between(ticks, values.min(axis=1), values.max(axis=1), color=color, alpha=.1)
                fgroup = [r for r in final if r["problem"] == kind and r["schedule"] == schedule
                          and r["method"] == method and r["beta"] == beta]
                summary.append(dict(problem=kind, schedule=schedule, method=method, beta=beta,
                                    mean_final_value=float(np.mean([r["final_value"] for r in fgroup])),
                                    mean_final_gradient_fro=float(np.mean([r["final_gradient_fro"] for r in fgroup])),
                                    mean_polar_error=float(np.mean([r["mean_polar_error"] for r in fgroup])),
                                    max_bound_ratio=max(r["bound_ratio"] for r in fgroup)))
            axes[i, 0].set(title=f"{schedule.capitalize()} step: {kind}",
                           ylabel="Objective gap" if kind == "quadratic" else "Objective value")
            axes[i, 1].set(title=f"{schedule.capitalize()} step: {kind}", ylabel="Current gradient Frobenius norm")
        for ax in axes.flat:
            ax.set_xlabel(f"Update count t (fixed T={horizon})")
            ax.grid(alpha=.2)
            ax.legend(fontsize=10)
        save_figure(fig, output, f"02_controlled_{kind}")
    return dict(runs=len(final), horizon=horizon, seeds=list(seeds),
                max_bound_ratio=max(r["bound_ratio"] for r in final),
                max_descent_residual=max(r["max_descent_residual"] for r in final),
                settings=summary)


def write_table(output, summary):
    lines = [r"\begin{tabular}{lllrr}", r"\toprule",
             r"目标 & 步长 & 方向与动量 & $f(W_T)$ & $\|G_T\|_\F$\\", r"\midrule"]
    for row in summary["settings"]:
        kind = "二次" if row["problem"] == "quadratic" else "非凸"
        schedule = "固定" if row["schedule"] == "constant" else "衰减"
        method = "SVD" if row["method"] == "svd" else "NS5"
        lines.append(f'{kind} & {schedule} & {method}, $\\beta={row["beta"]:g}$ & '
                     f'{row["mean_final_value"]:.6f} & {row["mean_final_gradient_fro"]:.6f}' + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "controlled_table.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results_v2"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=2048)
    args = parser.parse_args()
    if args.seeds < 1 or args.horizon < 1:
        parser.error("seeds 与 horizon 必须为正整数")
    # 不允许把新版实验输出写入原版结果目录。
    if args.output.resolve() == args.source.resolve():
        parser.error("output 须与原版 source 目录不同")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12})
    budget_plot(args.source, args.output)
    summary = controlled_experiment(args.output, range(args.seeds), args.horizon)
    summary["versions"] = dict(python=platform.python_version(), numpy=np.__version__,
                               matplotlib=matplotlib.__version__)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_table(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
