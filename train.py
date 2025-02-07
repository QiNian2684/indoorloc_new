# train.py

import torch
import numpy as np
from tqdm import tqdm
from utils import save_metrics, save_predictions, plot_training_curves, record_trial_result
import pandas as pd
import os

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs,
                early_stop_patience, result_dir, trial=None):
    """
    针对单次训练过程的封装。
    每次都在 result_dir 下保存最优模型、评估指标、可视化等。
    trial 用于 Optuna 若要支持中途 pruning 等。
    """
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    model.to(device)

    print(f"[INFO] Starting training for {num_epochs} epochs with early_stop_patience = {early_stop_patience}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Optuna 进行中途裁剪
        if trial is not None:
            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                raise Exception("Trial pruned")

        # 早停 + 保存最好模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(result_dir, "best_model.pth"))
        elif epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 评估阶段：加载最优模型，计算各种指标，保存
    model.load_state_dict(torch.load(os.path.join(result_dir, "best_model.pth")))
    model.eval()
    predictions_list = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    predictions = np.concatenate(predictions_list, axis=0)
    targets_arr = np.concatenate(targets_list, axis=0)

    mse = np.mean((predictions - targets_arr) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets_arr))
    horizontal_errors = np.sqrt(
        (predictions[:, 0] - targets_arr[:, 0]) ** 2 + (predictions[:, 1] - targets_arr[:, 1]) ** 2)
    avg_error = np.mean(horizontal_errors)
    median_error = np.median(horizontal_errors)

    metrics_dict = {
        "Best Epoch": best_epoch + 1,
        "Validation Loss": best_val_loss,
        "Test MSE": mse,
        "Test RMSE": rmse,
        "Test MAE": mae,
        "Average Horizontal Error (m)": avg_error,
        "Median Horizontal Error (m)": median_error
    }
    print(f"[INFO] 评价指标: {metrics_dict}")
    save_metrics(metrics_dict, result_dir)
    save_predictions(predictions, targets_arr, result_dir)

    # 记录高误差样本
    test_csv = "UJIndoorLoc/validationData.csv"
    try:
        test_data = pd.read_csv(test_csv)
        horizontal_err = np.sqrt(
            (predictions[:, 0] - targets_arr[:, 0]) ** 2 + (predictions[:, 1] - targets_arr[:, 1]) ** 2)
        threshold = 15.0
        high_error_indices = np.where(horizontal_err > threshold)[0]
        if len(high_error_indices) > 0:
            high_error_data = test_data.iloc[high_error_indices].copy()
            high_error_data['pred_x'] = predictions[high_error_indices, 0]
            high_error_data['pred_y'] = predictions[high_error_indices, 1]
            high_error_data['pred_floor'] = predictions[high_error_indices, 2]
            high_error_data['error_distance'] = horizontal_err[high_error_indices]
            high_error_csv = os.path.join(result_dir, "high_error_samples.csv")
            high_error_data.to_csv(high_error_csv, index=False)
            print(f"[INFO] 高误差样本已保存至 {high_error_csv}")
        else:
            print("[INFO] 没有发现高误差样本。")
    except Exception as e:
        print(f"[WARNING] 无法记录高误差样本：{e}")

    # 绘制训练曲线 & 记录结果
    trial_number = trial.number if trial is not None else -1
    trial_params = trial.params if trial is not None else {}
    plot_training_curves(
        train_losses,
        val_losses,
        trial_params,
        metrics_dict,
        predictions,
        targets_arr,
        result_dir,
        trial_number
    )

    if trial is not None:
        record_trial_result(trial, metrics_dict, result_dir)

    return best_val_loss, metrics_dict
