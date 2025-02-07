# optuna_tuner.py

import os
import optuna
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import UJIIndoorLocDataset
from model import IndoorLocalizationModel
from train import train_model
from utils import create_result_dir
import shutil

def objective(trial, main_result_dir):
    """
    Optuna 调用的目标函数。
    每次 trial 都会被调用一次，返回验证集上的最优损失。
    """

    # 1) 采样超参数
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.3, 0.7)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64, 128])
    early_stop_patience = trial.suggest_int('early_stop_patience', 3, 7)

    # 额外可调参数
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    cnn_channels_options = [
        (128, 256),
        (128, 256, 256),
        (128, 256, 512),
        (256, 256, 256),
    ]
    cnn_channels = trial.suggest_categorical('cnn_channels', cnn_channels_options)
    transformer_layers = trial.suggest_int('transformer_layers', 2, 6)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])

    num_epochs = 50

    # 2) 为当前 trial 创建单独的结果目录，形如 001, 002, ...
    #    trial.number 是从 0 开始的，这里为了呈现 3 位数格式，+1 再格式化
    trial_dir_name = f"{trial.number+1:03d}"
    trial_result_dir = os.path.join(main_result_dir, trial_dir_name)
    os.makedirs(trial_result_dir, exist_ok=True)

    # 打印当前超参数组合
    print(f"\n[INFO] Trial {trial.number} => Folder: {trial_dir_name}")
    print(f"  lr={lr:.6f}, dropout_rate={dropout_rate:.4f}, weight_decay={weight_decay:.6f}, "
          f"batch_size={batch_size}, early_stop_patience={early_stop_patience}, "
          f"embedding_dim={embedding_dim}, cnn_channels={cnn_channels}, "
          f"transformer_layers={transformer_layers}, nhead={nhead}, "
          f"dim_feedforward={dim_feedforward}")

    # 3) 数据加载
    train_csv = "UJIndoorLoc/trainingData.csv"
    test_csv = "UJIndoorLoc/validationData.csv"
    train_dataset = UJIIndoorLocDataset(train_csv, is_train=True)
    test_dataset = UJIIndoorLocDataset(test_csv, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4) 使用采样后的超参数初始化模型
    model = IndoorLocalizationModel(
        embedding_dim=embedding_dim,
        cnn_channels=cnn_channels,
        transformer_layers=transformer_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate
    )
    model.to(device)

    # 5) 优化器、调度器、损失等
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = torch.nn.MSELoss()

    # 6) 训练 & 返回验证集最优损失
    try:
        best_val_loss, metrics = train_model(
            model, train_loader, test_loader,
            criterion, optimizer, scheduler, device,
            num_epochs, early_stop_patience, trial_result_dir, trial
        )
    except Exception as e:
        if "Trial pruned" in str(e):
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    return best_val_loss

def run_optuna_study(n_trials=500):
    """
    运行 n_trials 次超参数搜索，并将最终最优 trial 的文件夹复制为 000。
    """
    # 创建带时间戳的总目录
    main_result_dir = create_result_dir()

    # 建立一个 Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, main_result_dir), n_trials=n_trials)

    # 搜索结束后，输出最佳 trial 信息
    print("\n[INFO] 所有超参数搜索完成。最佳 trial:")
    best_trial = study.best_trial
    print(f"  Value (Val Loss): {best_trial.value:.6f}")
    print("  超参数：")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 复制最佳 trial 的文件夹到 000
    best_trial_folder_name = f"{best_trial.number+1:03d}"  # trial.number + 1
    best_trial_folder_path = os.path.join(main_result_dir, best_trial_folder_name)
    copy_folder_path = os.path.join(main_result_dir, "000")
    if os.path.exists(copy_folder_path):
        shutil.rmtree(copy_folder_path)  # 若已存在则先删
    shutil.copytree(best_trial_folder_path, copy_folder_path)
    print(f"[INFO] 最佳 trial ({best_trial_folder_name}) 文件夹已复制为 {copy_folder_path}")

    # 可选：保存所有 trial 的简要结果表
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(main_result_dir, "optuna_study_results.csv"), index=False)
    print(f"[INFO] optuna_study_results.csv 已保存到 {main_result_dir}")

    return study, main_result_dir
