# 本次运行记录

环境：Python 3.12.14 / NumPy 2.3.5 / Matplotlib 3.10.8，CPU，float64。

执行命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m muon_study.experiments --output results --seeds 5
```

最终退出状态均为 0；11 项检查全部通过。
实验默认设置为 5 个种子、7 个预算、2 类目标、2 个动量设置，合计 140 个速率验证运行。
所有单次运行的平均梯度界及每一步下降界均通过；数值容限见脚本。

另有 20 个矩阵轨迹运行（5 个种子 × 4 个变体），检查含近似误差的事后界；
10 个神经网络训练运行（5 个种子 × 2 个方法）。
神经网络的梯度检查覆盖全部四个参数组。

第一次完整运行在汇总 JSON 写出时遇到 NumPy int64 序列化错误。
该处已显式转为 Python int，再完整运行成功。
两次完整运行产生的 6 个 CSV 文件用 SHA-256 比较，逐个完全一致。
运行时长和 PDF 的时间戳不作为确定性指标。

图表已人工查看；中文 LaTeX 报告已用 XeLaTeX 编译并检查页面、交叉引用与字体。
没有执行语言模型训练、分布式训练或 GPU 性能测试。
