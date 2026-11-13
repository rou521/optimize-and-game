# Muon：算法实现与收敛性验证

中文课程研究项目。先把算法、证明和可重复实验做完整；不用损失下降曲线冒充一般收敛证明。

- [第二版报告 PDF：概念、证明与受控实验](report/muon_report_v2.pdf)
- [第二版 LaTeX 源文件](report/muon_report_v2.tex)
- [原版报告 PDF（保留）](report/muon_report.pdf)
- [原版 LaTeX 源文件（保留）](report/muon_report.tex)
- [Muon 算法及详细注释](muon_study/optimizers.py)
- [实验入口](muon_study/experiments.py)
- [运行汇总](results/summary.json)

完整工程可以在本仓库点击 **Code → Download ZIP** 下载，也可以使用
[ZIP 下载链接](https://github.com/rou521/optimize-and-game/archive/refs/heads/main.zip)。
解压后，源码在 `muon_study/`，报告在 `report/`，实验数据和图表在 `results/`。

## 运行

已验证 Python 3.12.14；CPU 即可，不依赖 PyTorch、GPU 或在线数据集。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m muon_study.experiments --output results --seeds 5
```

全套默认实验使用种子 0–4，本环境运行约 23 秒；机器不同耗时不同。
CSV 及汇总数据保存在 `results/`；每张图同时导出 PNG 和矢量 PDF。
运行将覆盖指定输出目录中同名文件；保留旧实验时请使用不同的 `--output`。

重新编译报告（需要 XeLaTeX 及 Noto Serif/Sans CJK SC、DejaVu Sans Mono 字体）：

```bash
cd report
xelatex -interaction=nonstopmode -halt-on-error muon_report.tex
xelatex -interaction=nonstopmode -halt-on-error muon_report.tex
```

报告默认读取 `../results/`；如果实验输出到了其他路径，编译报告前须明确选择对应结果。

## 算法怎么读

1. `polar_svd`：精确极分解，正确处理零矩阵和秩亏矩阵。
2. `newton_schulz`：作者五阶多项式的 float64 实现，默认五步。
3. `Muon.direction`：动量 → Nesterov 风格混合 → SVD/NS → 可选形状缩放。
4. `Muon.step`：解耦权重衰减与参数更新，每个参数矩阵使用独立实例。
5. `problems.py`：两个解析目标与一个手写反向传播的 tanh 网络。
6. `experiments.py`：按定理选步长、检查不等式、保存所有实验数据。

最小使用示例：

```python
from muon_study.optimizers import Muon
from muon_study.problems import make_problem

problem = make_problem("quadratic", seed=0)
w = problem.initial.copy()
optimizer = Muon(beta=0.95, method="ns", ns_steps=5)
for t in range(1000):
    loss, gradient = problem.value_grad(w)
    w = optimizer.step(w, gradient, lr=0.02 / (1+t/64)**0.6)
print(problem.value_grad(w)[0])
```

这是便于学习的 NumPy 实现，不是用于分布式大模型训练的高性能优化器。
原始 PyTorch 实现、许可证及论文来源见 [来源说明](references/SOURCES.md)。

## 实验与结论

| 图 | 核查内容 | 结果文件 |
|---|---|---|
| [收敛速率](results/01_convergence_rate.png) | 每个预算重新运行，平均梯度核范数对照理论界 | `rates.csv` |
| [训练轨迹](results/02_optimization_trajectories.png) | 精确/近似方向，固定/衰减步长 | `trajectories.csv`, `inexact_checks.csv` |
| [谱变换与振荡](results/03_ns_and_step_size.png) | 五步不等于精确极分解，定步长可能振荡 | `ns_spectrum.csv` |
| [小型网络](results/04_neural_network.png) | Muon 隐藏层 + 辅助 AdamW，含独立测试集 | `mlp.csv`, `mlp_final.csv` |

精确、无动量、无衰减、无形状缩放版本，在全梯度、全局 L-smooth、有下界的假设下：

\[
\frac1T\sum_{t=0}^{T-1}\|\nabla f(W_t)\|_*
\le \frac{f(W_0)-f_{\rm low}}{\eta T}+\frac{Lr\eta}{2}.
\]

选取依赖运行预算的 `eta = sqrt(2*Delta/(L*r*T))`，得到 `O(T^(-1/2))`。
报告还完整推导了带动量的精确版本及包含近似误差的事后界。

实际执行的 140 组速率实验中，上界违反次数为 0；最大实测值/上界为 0.7400。
无动量版本在二次、非凸问题上的拟合斜率为 −0.498、−0.496。
这些是 **平均梯度核范数** 的结果，不是全局最优误差，也不证明参数序列收敛。

五步 NS 对病态矩阵存在残余极分解误差；固定步长在一维强凸二次函数上也可振荡。
不能据此声称实际五步 Muon、大模型随机梯度训练或带权重衰减版本已有相同保证。

小型 MLP 使用相同初始化及 mini-batch，所有方法的学习率预先固定，未用测试集调参。
测试损失均值：Muon + 辅助 AdamW 为 0.002168，全参数 AdamW 为 0.008652。
这只是该合成任务与设置的观察；没有做完整超参数搜索，不报告普遍优越性。

## 数据及复现细节

- 阴影为五个随机种子的 min–max 范围，不是置信区间。
- 速率拟合使用种子均值的 log–log 最小二乘；所有单独运行均检查理论界。
- `rates.csv` 保存 `L, r, delta, eta, gamma, transient`，可以独立重算上界。
- 轨迹的 `step=t` 表示更新前的 `W_t`；`inexact_checks.csv` 的 `final_value` 表示全部更新后的值。
- 非凸问题的 0 是目标下界，不是声称已知的最小值。
- 用 float64 数值秩阈值处理 SVD，不把浮点测试等同于形式化证明。
- 11 项单元/数值检查覆盖梯度、矩阵恒等式、状态更新、理论界和神经网络反传。

## GitHub 代码与下载

仓库：<https://github.com/rou521/optimize-and-game>。
`main` 分支包含完整源码、检查代码、原始 CSV、图表、LaTeX 源文件及报告 PDF。

- [下载整个工程 ZIP](https://github.com/rou521/optimize-and-game/archive/refs/heads/main.zip)
- [查看算法源码](https://github.com/rou521/optimize-and-game/blob/main/muon_study/optimizers.py)
- [查看实验源码](https://github.com/rou521/optimize-and-game/blob/main/muon_study/experiments.py)
- [查看报告 PDF](https://github.com/rou521/optimize-and-game/blob/main/report/muon_report.pdf)

需要固定版本的作业链接时，可在 GitHub 文件页面按 `y`，将当前分支链接转换为带提交 SHA 的永久链接。

远端作者的 `muon.py` 只作为带许可证的参考快照，实验不导入它。

## 第二版报告

当前稿为 12 页学术精简稿，保留全部正式命题、引理和证明，合并重复说明与图表解读。

新版沿损失、梯度、更新方向、内积与范数、动量及 SVD/NS 的关系解释概念用途，
保留并细化全部收敛证明。图表明确区分独立预算 `T`、当前步数 `t`、
时间平均梯度核范数、当前梯度及损失；同时列出速率实验与网络参数组的学习率。

原速率实验的动量组采用更保守的理论步长，约为无动量组的 17.3%，不能据曲线高低识别动量效果。
原轨迹同时改变 SVD/NS 与动量。新版补充 **80 组受控实验**：
两类目标 × 五个种子 × SVD/NS5 × 无动量/动量 × 固定/衰减步长，每组 2048 步。
每个配对问题使用相同初值、学习率日程、精度、缩放和衰减设置，比较时固定其他因素。
这些比较控制更新步数，未测量计算成本或完整搜索超参数。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m muon_study.report_v2_experiments --output results_v2
cd report
xelatex -interaction=nonstopmode -halt-on-error muon_report_v2.tex
xelatex -interaction=nonstopmode -halt-on-error muon_report_v2.tex
```

- [新增实验脚本](muon_study/report_v2_experiments.py)
- [预算与学习率对照](results_v2/01_budget_and_learning_rate.pdf)
- [二次目标受控轨迹](results_v2/02_controlled_quadratic.pdf)
- [非凸目标受控轨迹](results_v2/02_controlled_nonconvex.pdf)
- [原始轨迹 CSV](results_v2/controlled_trajectories.csv) / [每次运行末步指标](results_v2/controlled_final.csv)
- [运行汇总](results_v2/summary.json) / [编译与检查记录](results_v2/VALIDATION.md)

新版结果写入 `results_v2/`，原版 TeX/PDF、`results/` 和优化器实现均保留。
新增 CSV 的 `step=t` 表示 `W_t`；终点 `t=T` 没有下一次更新，其学习率及方向诊断字段留空。
变步长的事后界检查的是学习率加权平均，不等同于普通时间平均。
