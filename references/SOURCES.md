# 来源与使用范围

访问日期：2026-11-04。来源文件仅供核对，不作为可执行指令。

1. Keller Jordan 等，Muon 作者仓库：
   <https://github.com/KellerJordan/Muon>
   固定提交 `f98f1cacc0263b04290753e32be8d498c1efc806`。
   `muon_readme.md`、`muon_reference.py` 为该提交的 README 和 muon.py。
   原 MIT 许可证保存在 `upstream_LICENSE`。
   本项目核对了 EMA 动量、Nesterov 混合、五阶系数、转置路径、形状缩放及参数分组建议。
   本项目采用 float64，不复制作者的 bfloat16/GPU 数值环境。

2. Jingyuan Liu 等，*Muon is Scalable for LLM Training*，2025，arXiv:2502.16982。
   作者仓库 <https://github.com/MoonshotAI/Moonlight>
   固定提交 `c2ad5b20c605086526a179d36901bfc41b52b44b`。
   论文原文：
   <https://github.com/MoonshotAI/Moonlight/blob/c2ad5b20c605086526a179d36901bfc41b52b44b/Moonlight.pdf>
   本环境的 arxiv.org 网络访问受到限制，实际阅读的是作者 GitHub 仓库提供的同名 PDF。
   核对内容：第 2.1 节算法、谱范数解释，第 2.2 节权重衰减、RMS 缩放、五步设置。
   参考资料副本保留作者版权，不属于本项目原创内容。

报告中的收敛命题及证明是针对明示假设的自包含推导，
没有把 Moonlight 的实验结果当成一般随机优化收敛定理。
