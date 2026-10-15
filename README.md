# 实验一：十类文本分类

代码使用 scikit-learn 比较朴素贝叶斯、逻辑分类、线性 SVM 与 MLP，并保存每轮损失。最终预测方案为 TF-IDF + 参数平均的 SGD 线性 SVM。无标签测试集只在最终配置锁定后读取；本地报告的 0.944440 是验证集 Macro-F1，不是测试集分数。

## 环境与数据

- Python 3.10；依赖版本见 `requirements.txt`。
- 将实验提供的原始 `train_data.csv` 和 `test_data_unlabeled.csv` 放入本目录的 `data/`。不要更改行顺序或内容。仓库不包含课程数据；如果数据在别处，可设置环境变量 `EXPERIMENT1_DATA_DIR` 指向所在目录。
- 训练脚本会校验原始数据及固定划分的 SHA-256；数据版本不符时会中止，不会悄悄重新划分。

在 PowerShell 中运行：

```powershell
cd "本仓库路径"
python -m pip install -r requirements.txt
$env:EXPERIMENT1_DATA_DIR = "原始CSV所在目录"  # 已放入 data/ 时可省略
python -B -u compare_models.py --check-only
python -B -u finalize_smooth_sgd_svm.py --check
```

## 复现实验与最终预测

`compare_models.py` 比较四类模型和预设的参数网格；其线性模型使用自动调整学习率。朴素贝叶斯是一次拟合，不用 SGD。MLP 的完整网格约需较长时间。

```powershell
python -B -u compare_models.py --linear-max-epochs 80 --mlp-max-epochs 120
python -B -u compare_sgd_schedules.py --epochs 120 --only constant_2_averaged
python -B -u finalize_smooth_sgd_svm.py --validate
```

报告中的附加对照可分别用 `ablate_preprocessing_sgd.py`（固定四轮比较文本处理）和 `compare_mlp_architectures.py`（只改变隐藏层结构）复现。后者会重新训练三组 MLP，耗时较长；可先运行 `python -B -u compare_mlp_architectures.py --check-only` 核对数据与划分。参数扫描与最终模型使用不同的学习率设置，不能把扫描时的最好分数写成最终预测模型的分数。

报告中另外两组历史验证对照——原始文本 5000 维 TF-IDF 加 `MultinomialNB(alpha=0.1)`（F1 约 0.8922），以及早期 `learning_rate='optimal'` 的线性 SVM 三组 `alpha`（各最多 20 轮，分别按验证 Macro-F1 选轮）——使用独立入口复现。它只读取带标签训练 CSV，不生成提交预测文件；结果保存到新建的 `report_baseline_reproduction/` 时间戳目录。先检查，再运行：

```powershell
python -B -u reproduce_report_baselines.py --check-only
python -B -u reproduce_report_baselines.py --models nb svm
```

该早期 SVM 扫描与最终采用的恒定学习率、参数平均、第 70 轮模型不是同一次训练；报告不能将二者分数混用。

锁定配置的划分敏感性检查固定另外五个分层随机划分种子 43–47，**每次独立在该训练子集拟合 TF-IDF**；SVM 始终使用 `alpha=10⁻⁴`、恒定学习率 `eta0=2.0`、参数平均和 70 轮，不根据这五次的结果重新选参或选轮。该脚本不读取无标签测试集，结果写入新建的 `split_stability_checks/` 时间戳目录。种子 42 的开发结果不参与这五次的均值和标准差。

```powershell
python -B -u check_split_stability.py --check-only
python -B -u check_split_stability.py
```

前面的 python -B -u finalize_smooth_sgd_svm.py --validate 命令会打印 Validation directory:。确认该次验证的 Macro-F1 约为 0.944440、保存轮次为 70，再使用这次验证输出的目录进行全量重训：

```powershell
$validationDir = "--validate 命令打印的 Validation directory 完整路径"
python -B -u finalize_smooth_sgd_svm.py --finalize --validation-dir $validationDir
```

最终生成的 `full_refit_*/predictions.csv` 为 2457 行、单列、无表头的类别编号。提交时将它复制并命名为 `prediction.csv`。不要将验证结果说成测试集准确率；提供的测试 CSV 没有标签。

## 报告图与结果证据

`results_summary/` 是适合放入 GitHub 的精简结果快照，不包含课程原始数据或大型模型文件。其 [文件说明](results_summary/README.md) 标出每张表对应的实验脚本。运行完整训练后，生成的时间戳目录仍留在本机；这些目录被 `.gitignore` 排除，避免把模型二进制和大量重复日志上传。

报告中的三模型训练／验证损失图可**仅凭仓库内的逐轮 CSV** 重画，无需重新训练，也无需读取无标签测试集：

```powershell
python -B plot_training_curves.py --check-only
python -B plot_training_curves.py
```

第二条命令生成 `results_summary/loss_curves.png`，虚线分别标出逻辑分类第 20 轮、线性 SVM 第 70 轮、MLP 第 119 轮。图中逻辑分类与 MLP 使用多类对数损失，SVM 使用 one-vs-rest 平均 hinge 损失；不同损失的绝对值不可直接比较。

报告各项结果可据此核对：最终四模型结果参考 `parameter_comparison.csv`、`schedule_comparison.csv`、`final_selection.json` 和 `svm_epoch_history.csv`；旧 5000 维朴素贝叶斯基线见 `baseline_reproduction.json`，早期 SVM 三组参数见 `early_svm_comparison.csv`；文本消融、MLP 结构对照及五种子检查分别见 `preprocessing_comparison.csv`、`mlp_architecture_comparison.csv` 和 `split_stability.csv`。注意早期线性模型与最终恒定学习率方案是不同训练配置，不应把它们的分数混作一次公平对照。
