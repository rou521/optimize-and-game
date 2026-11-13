# 第二版编译与检查记录

数值实验记录：2026-11-11；文字精修与重新编译：2026-11-13（Asia/Shanghai）。基于原提交 `ae64be81b4b19647d38c36a5e7ad4012dba6cf8b`。

## 保留原版

原版 TeX/PDF、原 `results/`、优化器与原实验脚本均未改动。
逐一对照原 `SHA256SUMS.json` 验证全部原归档文件（仅 README 增加新版入口）。
本次提交另更新校验清单以覆盖新版文件。

## 运行与数值检查

环境：Python 3.12.14、NumPy 2.3.5、Matplotlib 3.10.8，CPU/float64。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m muon_study.experiments --output /tmp/muon-v1-recheck --seeds 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m muon_study.report_v2_experiments --output results_v2
```

- 11 项原有检查全部通过。
- 原实验在独立目录完整复跑，6 个 CSV 与 `results/` 逐字节相同：`rates.csv`、`trajectories.csv`、`inexact_checks.csv`、`ns_spectrum.csv`、`mlp.csv`、`mlp_final.csv`。原 140 组理论界验证均通过。
- 新增 80 组运行：两类目标 × 五个种子 × 两种方法 × 两种动量 × 两种步长日程，每组 2048 步。
- `controlled_trajectories.csv` 共 20560 行数据：每组 257 行，记录 `t=0,8,...,2048`。
- 独立核查所有配对组的初始目标、初始梯度及初始学习率一致；每条记录的学习率与指定日程一致。
- 独立核查末行确为 `W_T`，与 `controlled_final.csv` 精确相同；末行不含额外更新，其方向诊断字段留空。
- 对每组逐步检查含动量及近似误差的下降界，并检查学习率加权事后界。最大实测/上界为 0.957137070105；最大逐步残差为 -4.969716606568e-05（非正）。
- 独立重算全部 16 个汇总设置的五种子均值，检查 JSON 与末步 CSV 一致。LaTeX 表格由同次运行自动生成。
- 更新图表字号后重新生成，数值与第一次 v2 运行一致；没有重新调参或根据测试集选取方案。

## PDF 编译与页面检查

编译器：XeTeX 3.141592653-2.6-0.999996 (TeX Live 2025/dev/Debian)。使用已有 Noto CJK 与 DejaVu 字体。

```bash
cd report
XDG_CACHE_HOME=/tmp/muon-xdg-cache xelatex -interaction=nonstopmode -halt-on-error muon_report_v2.tex
XDG_CACHE_HOME=/tmp/muon-xdg-cache xelatex -interaction=nonstopmode -halt-on-error muon_report_v2.tex
```

两次最终编译退出码均为 0，PDF 共 12 页。
最终日志无 LaTeX 警告、未定义交叉引用/引用、缺失字符、Overfull 或 Underfull 提示。
使用 PDF 文本及边界检查确认无 `??` 引用、无页面外文本/图片块；`pdffonts` 确认字体全部嵌入。
查看全部页面缩略图，并放大检查证明、预算与学习率图、受控图表、末步表、复现命令及参考文献。
已检查复现命令、图表说明与参考文献的排版。

## 2026-11-13 学术语言精修

- 将问答式标题与口语化说明改为学术表述，压缩摘要、概念解释、重复结论及操作说明。
- 删除已由受控对照覆盖的旧轨迹图，保留预算图、两类目标的受控轨迹、NS 谱实验及网络结果；图表就近配置分析。
- 页数由 17 页减为 12 页，字号、行距与页边距未变。
- 与上一提交逐段比对，2 个命题、2 个引理及 5 段正式证明的 LaTeX 内容完全一致；下降引理的积分推导亦保留。
- 本次为报告编辑，未重新运行数值实验。优化器、实验脚本、CSV、图像和汇总数据均与上一提交的校验值一致。
- XeLaTeX 重新编译两次，检查引用、字符、溢出、页面边界与图文衔接，最终日志无警告。

## 文件 SHA-256

- `report/muon_report.tex`：`a6b3d4c6b98bd95a6bb43d91f89d1fd02fbdb6bab200cd4e80c6bc5f42a162bc`
- `report/muon_report.pdf`：`e47b0209070f005227b5ab9afd0b9850f26b690ec19e0ff37f900fb964a2af99`
- `report/muon_report_v2.tex`：`ec1857ccedf8b52ab87e0093e34b0ef53fc9d79680230c355b6706ab0cd48ff6`
- `report/muon_report_v2.pdf`：`680ad5b98b04694324281066dcbad010ee03a1e27969faeec410a5aeebc655f5`

完整项目校验值见根目录 `SHA256SUMS.json`（清单不包含自身）。

## 结论边界

共同学习率对照控制更新次数，不控制真实运行时间；未完整搜索超参数。
原网络对比保留为两套训练方案示例，其隐藏层学习率不同，不作为动量或 NS 的单因素证据。
浮点界检查不替代解析证明；NS5 仍保留近似误差项。
