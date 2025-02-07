# utils.py

import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import shutil

def create_result_dir(root_dir='result'):
    """
    在项目根目录下创建一个以当前时间戳命名的结果文件夹
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
    保存测试集上每个样本的预测坐标、原始坐标以及差距
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

def plot_training_curves(train_losses, val_losses, trial_params, final_metrics, result_dir, trial_number):
    """
    绘制训练与验证损失的折线图，并保存图像。
    图中标题包含当前 trial 的超参数及最终评价指标。
    图片文件名使用三位数编号，如 001.png, 002.png, 等。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title_text = f"Trial {trial_number:03d}\nParams: {trial_params}\nFinal Metrics: {final_metrics}"
    plt.title(title_text)
    plt.legend()
    image_path = os.path.join(result_dir, f"{trial_number:03d}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"[INFO] 训练曲线图已保存至 {image_path}")

def copy_best_trial_image(best_trial_number, result_dir):
    """
    将最佳 trial 的图片复制为 000.png
    """
    best_image = os.path.join(result_dir, f"{best_trial_number:03d}.png")
    best_copy = os.path.join(result_dir, "000.png")
    shutil.copy(best_image, best_copy)
    print(f"[INFO] 最佳 trial 的图片已复制为 {best_copy}")
