### 项目简介

这是一个基于 **UJIIndoorLoc** 数据集的室内定位项目，使用 PyTorch 实现。模型是 CNN + Transformer 的混合架构，可对输入的 WiFi RSSI 信号进行回归预测，输出经度 (x)、纬度 (y) 和楼层 (floor)。项目支持**两种模式**：

1. **训练模式（`--mode train`）**：使用固定超参数对模型进行训练。  
2. **调参模式（`--mode tune`）**：使用 [Optuna](https://optuna.org/) 对多个超参数进行自动搜索与优化。

### 目录结构

```
.
├── data.py                # 数据加载及预处理
├── main.py                # 主入口脚本，可选择 train / tune 模式
├── model.py               # 模型定义（可配置多层 CNN 与 Transformer）
├── optuna_tuner.py        # 调参脚本，定义超参数搜索空间及 objective 函数
├── train.py               # 训练过程、验证评估、早停等逻辑
├── utils.py               # 工具函数，如日志、可视化、文件保存等
└── UJIndoorLoc
    ├── trainingData.csv   # 训练数据
    └── validationData.csv # 验证（测试）数据
```

### 环境依赖

- Python 3.x  
- PyTorch >= 1.7  
- optuna >= 2.0  
- pandas  
- numpy  
- matplotlib  
- tqdm  

可使用以下命令安装常用依赖：  
```bash
pip install torch optuna pandas numpy matplotlib tqdm
```

### 数据准备

确保在项目根目录下有 `UJIndoorLoc` 文件夹，内部包含以下 CSV 文件：

- `trainingData.csv`：训练集  
- `validationData.csv`：验证/测试集  

在原始的 [UJIIndoorLoc](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc) 数据中，RSSI 列通常以 `WAP001`, `WAP002`, ... 的格式命名，共有 520 列。该项目脚本中会自动检测以 `WAP` 开头的列，并进行归一化处理和缺失值替换。

如果你的数据列名或格式和本项目不一致，需要在 `data.py` 中相应修改列的检测逻辑。

### 使用方法

1. **训练模式**  
   ```bash
   python main.py --mode train \
                  --batch_size 64 \
                  --early_stop_patience 5
   ```
   - `--batch_size`: 指定批大小，若不写则默认 64。  
   - `--early_stop_patience`: 提前停止的耐心值（验证集若在若干个 epoch 内无改善则停止），若不写则默认 5。  

   训练完成后，会在 `result/XXXXXX`（时间戳命名）文件夹下保存：  
   - `best_model.pth`：验证集上最优权重  
   - `metrics.csv` / `predictions.csv` / 若干 PNG 图等

2. **调参模式**  
   ```bash
   python main.py --mode tune --n_trials 50
   ```
   - `--n_trials`: 指定要尝试的搜索次数（默认值可能写在脚本里是 500，但你可以根据需要自行指定）。  

   调参过程结束后，会输出：  
   - `optuna_study_results.csv`：所有 trial 的结果  
   - `trial_results.csv`：每次 trial 的详细指标  
   - `xxx.png`：不同 trial 的训练曲线、误差散点可视化  
   - `000.png`：最佳 trial 的图像副本

### 常见问题

1. **为什么需要两种模式？**  
   - 当已经对模型结构和超参数比较确定时，使用**训练模式**快速运行即可；当需要自动搜索更多超参数以期获得更优解时，则使用**调参模式**。  

2. **训练速度慢怎么办？**  
   - 模型非常大时，可以先把 `n_trials` 设小一些（例如 10 或 20）跑通流程；  
   - 或者在 `train.py` 中缩短 `num_epochs`；  
   - 如果有 GPU，则确保安装了 CUDA 版本 PyTorch 并在训练时使用 `cuda`。  

3. **数据列名不一致时如何修改？**  
   - 在 `data.py` 中修改提取 RSSI 列和目标列的逻辑，如 `self.rssi_columns`、`self.target_columns` 等。  

4. **怎么使用最优的 trial 进行最终训练？**  
   - 在 `optuna_study_results.csv` 或控制台中找到最佳 trial 的超参数组合，然后在 `main.py`（或你单独写的脚本）里手动配置这些参数，执行一次**训练模式**跑到更多 epoch 即可得到最终模型。  

----

如上所述，即为本项目的基本使用说明。如果还有任何疑问，可以在对应脚本中查看注释，或者在 issue 中进行讨论。祝您使用愉快！
