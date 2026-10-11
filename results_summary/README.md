# 报告结果的精简证据

这些 CSV 是已完成运行的原样快照；原始课程数据、大型模型文件和完整运行目录不放进仓库。数值均为验证集结果，不是无标签测试集成绩。固定数据划分种子为 42；划分敏感性检查另外使用 43–47。

| 文件 | 来源脚本与用途 |
| --- | --- |
| `logistic_epoch_history.csv` | `compare_sgd_schedules.py` 的 `constant_2_averaged` 逻辑分类，报告第 20 轮曲线 |
| `svm_epoch_history.csv` | `finalize_smooth_sgd_svm.py --validate`，报告最终 SVM 第 70 轮曲线 |
| `mlp_epoch_history.csv` | `compare_models.py` 的 `MLP_SGD_eta0.01`，报告第 119 轮曲线 |
| `parameter_comparison.csv` | `compare_models.py`，各模型参数对照 |
| `early_svm_comparison.csv` | `reproduce_report_baselines.py`，早期 SVM 参数比较 |
| `schedule_comparison.csv` | `compare_sgd_schedules.py`，恒定学习率加参数平均对照 |
| `mlp_architecture_comparison.csv` | `compare_mlp_architectures.py`，隐藏层结构对照 |
| `preprocessing_comparison.csv` | `ablate_preprocessing_sgd.py`，文本处理消融 |
| `split_stability.csv` | `check_split_stability.py`，种子 43–47 的划分检查 |
| `baseline_reproduction.json` | `reproduce_report_baselines.py`，5000 维朴素贝叶斯旧基线 |
| `final_protocol.json`、`final_selection.json` | 最终超参数、数据指纹、选中轮次与验证分数 |

运行 `python -B plot_training_curves.py` 可从逐轮 CSV 重画训练／验证损失图，生成 `loss_curves.png`。三种模型的损失定义不同，只比较各自曲线的变化趋势，不比较绝对数值。
