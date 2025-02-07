# train.py

import torch
import numpy as np
from tqdm import tqdm
from utils import save_metrics, save_predictions, plot_training_curves


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs,
                early_stop_patience, result_dir, trial=None):
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    model.to(device)

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

        # 验证阶段，使用测试集作为验证集
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

        # 调整学习率
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # 如果传入 trial，则向 Optuna 报告中间结果
        if trial is not None:
            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                raise Exception("Trial pruned")

        # Early Stopping 策略
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            # 保存当前最佳模型
            best_model_state = model.state_dict()
            torch.save(best_model_state, f"{result_dir}/best_model.pth")
        elif epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型进行测试评估
    model.load_state_dict(torch.load(f"{result_dir}/best_model.pth"))
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets_arr = np.concatenate(targets_list, axis=0)

    # 计算评价指标
    mse = np.mean((predictions - targets_arr) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets_arr))
    # 计算水平定位误差（仅对 x, y 坐标计算欧氏距离）
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

    # 绘制训练曲线图，包含超参数和最终评价指标
    trial_number = trial.number if trial is not None else -1
    trial_params = trial.params if trial is not None else {}
    plot_training_curves(train_losses, val_losses, trial_params, metrics_dict, result_dir, trial_number)

    return best_val_loss, metrics_dict
