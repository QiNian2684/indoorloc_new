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
from utils import create_result_dir, copy_best_trial_image

def objective(trial, result_dir):
    # 原有超参数采样
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.3, 0.7)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64, 128])
    early_stop_patience = trial.suggest_int('early_stop_patience', 3, 7)

    # 新增可调参数
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    # CNN 通道配置可以有多种方案
    cnn_channels_options = [
        (128, 256),
        (128, 256, 256),
        (128, 256, 512),
        (256, 256, 256)
    ]
    cnn_channels = trial.suggest_categorical('cnn_channels', cnn_channels_options)
    transformer_layers = trial.suggest_int('transformer_layers', 2, 6)
    nhead = trial.suggest_categorical('nhead', [2, 4, 8])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])

    num_epochs = 50

    # 打印当前超参数组合
    print(f"[INFO] Trial {trial.number}: "
          f"lr={lr:.6f}, dropout_rate={dropout_rate:.4f}, weight_decay={weight_decay:.6f}, "
          f"batch_size={batch_size}, early_stop_patience={early_stop_patience}, "
          f"embedding_dim={embedding_dim}, cnn_channels={cnn_channels}, "
          f"transformer_layers={transformer_layers}, nhead={nhead}, "
          f"dim_feedforward={dim_feedforward}")

    # 加载训练集与测试集
    train_csv = "UJIndoorLoc/trainingData.csv"
    test_csv = "UJIndoorLoc/validationData.csv"
    train_dataset = UJIIndoorLocDataset(train_csv, is_train=True)
    test_dataset = UJIIndoorLocDataset(test_csv, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建复杂模型
    model = IndoorLocalizationModel(
        embedding_dim=embedding_dim,
        cnn_channels=cnn_channels,
        transformer_layers=transformer_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout_rate=dropout_rate
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = torch.nn.MSELoss()

    try:
        best_val_loss, metrics = train_model(
            model, train_loader, test_loader,
            criterion, optimizer, scheduler, device,
            num_epochs, early_stop_patience, result_dir, trial
        )
    except Exception as e:
        if "Trial pruned" in str(e):
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    return best_val_loss

def run_optuna_study(n_trials=500):
    result_dir = create_result_dir()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, result_dir), n_trials=n_trials)

    print("最佳 trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  超参数：")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 将最佳 trial 的图片复制为 000.png
    best_trial_number = best_trial.number
    copy_best_trial_image(best_trial_number, result_dir)

    # 保存所有 trial 结果
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(result_dir, "optuna_study_results.csv"), index=False)
    print("[INFO] 超参数调优完成。")
    return study, result_dir
