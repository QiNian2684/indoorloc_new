# main.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
from model import IndoorLocalizationModel
from data import UJIIndoorLocDataset
from train import train_model
from optuna_tuner import run_optuna_study
from utils import create_result_dir
import pandas as pd


def main(args):
    # 列出项目根目录的所有文件
    print("[INFO] 项目根目录文件列表：")
    for f in os.listdir('.'):
        print(f"  {f}")

    if args.mode == 'tune':
        # 调参模式：运行 500 次 Optuna 试验
        print("[INFO] 开始使用 Optuna 进行超参数调优...")
        study, result_dir = run_optuna_study(n_trials=500)
    elif args.mode == 'train':
        # 最终训练模式，使用指定超参数（可替换为调参获得的最佳超参数）
        result_dir = create_result_dir()
        lr = 1e-3
        dropout_rate = 0.5
        weight_decay = 1e-5
        batch_size = 64
        num_epochs = 50
        early_stop_patience = 5

        # 加载训练集和测试集（已分好）
        train_csv = "UJIndoorLoc/trainingData.csv"
        test_csv = "UJIndoorLoc/validationData.csv"
        train_dataset = UJIIndoorLocDataset(train_csv, is_train=True)
        test_dataset = UJIIndoorLocDataset(test_csv, is_train=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IndoorLocalizationModel(dropout_rate=dropout_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                               verbose=True)
        criterion = torch.nn.MSELoss()

        print("[INFO] 开始最终训练...")
        best_val_loss, metrics = train_model(
            model, train_loader, test_loader,
            criterion, optimizer, scheduler, device,
            num_epochs, early_stop_patience, result_dir
        )
        # 保存最终评价指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(result_dir, "final_metrics.csv"), index=False)
        print("[INFO] 最终训练完成。评价指标如下：")
        print(metrics)
    else:
        print("无效的 mode，请选择 'train' 或 'tune'。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UJIIndoorLoc 三坐标回归模型训练（包含楼层回归）")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'],
                        help="模式：'train' 为最终训练，'tune' 为超参数调优")
    parser.add_argument('--n_trials', type=int, default=500,
                        help="超参数调优时的试验次数（mode 为 'tune' 时有效）")
    args = parser.parse_args()
    main(args)
