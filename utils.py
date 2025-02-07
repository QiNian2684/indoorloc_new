# utils.py

import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import csv
import numpy as np

def create_result_dir(root_dir='result'):
    """
    在项目根目录下创建以当前时间戳命名的结果文件夹
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(root_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    print(f"[INFO] 创建结果文件夹：{result_dir}")
    return result_dir

def save_metrics(metrics_dict, result_dir, filename='metrics.csv'):
    """
    将评价指标保存为 CSV 文件
    """
    df = pd.DataFrame([metrics_dict])
    file_path = os.path.join(result_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"[INFO] 评价指标已保存至 {file_path}")

def save_predictions(predictions, targets, result_dir, filename='predictions.csv'):
    """
    保存测试集上每个样本的预测与真实坐标（包括中间误差）
    """
    diffs = predictions - targets
    df = pd.DataFrame({
        'pred_x': predictions[:, 0],
        'pred_y': predictions[:, 1],
        'pred_floor': predictions[:, 2],
        'true_x': targets[:, 0],
        'true_y': targets[:, 1],
        'true_floor': targets[:, 2],
        'diff_x': diffs[:, 0],
        'diff_y': diffs[:, 1],
        'diff_floor': diffs[:, 2]
    })
    file_path = os.path.join(result_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"[INFO] 样本预测结果已保存至 {file_path}")

def write_csv_row(csv_file_path, fieldnames, row_data):
    """
    写入一行数据到 CSV 文件（若文件不存在则创建并写入表头）
    """
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f"[INFO] 记录已写入 {csv_file_path}")

def record_trial_result(trial, final_metrics, result_dir, csv_filename="trial_results.csv"):
    """
    将每个 trial 的超参数与评价指标记录到 CSV 文件中
    """
    csv_path = os.path.join(result_dir, csv_filename)
    row_data = {"Trial": trial.number}
    for k, v in trial.params.items():
        row_data[k] = v
    for k, v in final_metrics.items():
        row_data[k] = v
    fieldnames = list(row_data.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
    print(f"[INFO] Trial {trial.number} 结果已记录到 {csv_path}")

def plot_training_curves(train_losses, val_losses, trial_params, final_metrics, predictions, targets, result_dir,
                         trial_number):
    """
    绘制包含三部分的图像：
      1. 左上角：2D 预测误差散点图（经度、纬度误差及色条表示误差距离）
      2. 右上角：超参数和最终评价指标的文本显示
      3. 下方：训练与验证损失曲线
    图片保存为三位编号，如 001.png
    """
    plt.rcParams['font.family'] = 'Times New Roman'

    # 计算 2D 误差（仅针对经纬度）
    error_x = predictions[:, 0] - targets[:, 0]
    error_y = predictions[:, 1] - targets[:, 1]
    error_distance = np.sqrt(error_x ** 2 + error_y ** 2)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # 子图 1：误差散点图
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(error_x, error_y, c=error_distance, cmap='viridis', alpha=0.6)
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label('Error Distance (m)')
    ax1.set_title('2D Prediction Errors')
    ax1.set_xlabel('Longitude Error (m)')
    ax1.set_ylabel('Latitude Error (m)')

    # 子图 2：超参数与评价指标文本
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    params_text = "Training Parameters:\n"
    for k, v in trial_params.items():
        params_text += f"{k}: {v}\n"
    metrics_text = "\nFinal Metrics:\n"
    for k, v in final_metrics.items():
        if isinstance(v, float):
            metrics_text += f"{k}: {v:.4f}\n"
        else:
            metrics_text += f"{k}: {v}\n"
    combined_text = params_text + metrics_text
    ax2.text(0.5, 0.5, combined_text, fontsize=12, ha='center', va='center', wrap=True)

    # 子图 3：损失曲线
    ax3 = fig.add_subplot(gs[1, :])
    epochs_range = range(1, len(train_losses) + 1)
    ax3.plot(epochs_range, train_losses, 'r-', label='Train Loss')
    ax3.plot(epochs_range, val_losses, 'b-', label='Validation Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training and Validation Loss')
    ax3.legend()

    image_path = os.path.join(result_dir, f"{trial_number:03d}.png")
    plt.savefig(image_path)
    plt.close(fig)
    print(f"[INFO] 训练曲线及评估图已保存至 {image_path}")

def copy_best_trial_image(best_trial_number, result_dir):
    """
    将最佳 trial 的图片复制为 000.png
    """
    best_image = os.path.join(result_dir, f"{best_trial_number:03d}.png")
    best_copy = os.path.join(result_dir, "000.png")
    shutil.copy(best_image, best_copy)
    print(f"[INFO] 最佳 trial 的图片已复制为 {best_copy}")
