import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================

INPUT_DIR = "数据分析"
OUTPUT_DIR = "输出报告"
FILE_PATTERN = "*_comments_with_dimensions_*.csv"

PREDICT_DAYS = 30
WARNING_THRESHOLD = 0.45


# ==================== 核心函数 ====================

def prepare_data(df):
    """准备时间序列数据"""
    daily = df.groupby("date").agg({"情感得分": "mean"}).reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily.sort_values("ds").reset_index(drop=True)


def simple_forecast(daily_data, periods):
    """
    使用加权移动平均法进行简单预测
    """
    recent = daily_data.tail(30).copy()
    recent["ma7"] = recent["y"].rolling(7, min_periods=3).mean()

    base_value = recent["ma7"].iloc[-1]
    if pd.isna(base_value):
        base_value = recent["y"].iloc[-1]

    std = recent["y"].std()

    last_date = daily_data["ds"].max()
    pred_dates = [last_date + timedelta(days=i + 1) for i in range(periods)]

    np.random.seed(42)
    noise = np.random.normal(0, std * 0.3, periods)
    pred_values = base_value + noise
    pred_values = np.clip(pred_values, 0, 1)

    pred_df = pd.DataFrame({
        'ds': pred_dates,
        '预测得分': pred_values,
        '下限': pred_values - std * 0.5,
        '上限': pred_values + std * 0.5
    })
    pred_df['预警'] = pred_df['预测得分'] < WARNING_THRESHOLD

    return pred_df


def process_scenic(file_path):
    """处理单个景区"""
    basename = os.path.basename(file_path)
    scenic_name = basename.split("_")[0]

    print(f"\n处理: {scenic_name}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  数据: {len(df)} 条")

    daily_data = prepare_data(df)
    if len(daily_data) < 10:
        print(f"  跳过: 数据不足10天")
        return None

    print(f"  生成 {PREDICT_DAYS} 天预测...")
    pred_df = simple_forecast(daily_data, PREDICT_DAYS)

    return scenic_name, daily_data, pred_df


def generate_forecast_report(scenic_name, daily_data, pred_df):
    """生成预测报告"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"情感得分预测预警报告 - {scenic_name}")
    lines.append("=" * 60)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"预测天数: {PREDICT_DAYS}")
    lines.append(f"预警阈值: {WARNING_THRESHOLD}")
    lines.append(f"历史数据: {daily_data['ds'].min().date()} 至 {daily_data['ds'].max().date()}")

    warning_count = pred_df["预警"].sum()
    lines.append(f"预测期间预警天数: {warning_count}")

    if warning_count > 0:
        lines.append("\n预警日期:")
        wdf = pred_df[pred_df["预警"]]
        for _, row in wdf.iterrows():
            lines.append(f"  {row['ds'].strftime('%Y-%m-%d')}: 预测得分 {row['预测得分']:.3f}")
    else:
        lines.append("\n预测期间无预警")

    lines.append("")
    lines.append("-" * 40)
    lines.append("预测统计:")
    lines.append(f"  最高: {pred_df['预测得分'].max():.4f}")
    lines.append(f"  最低: {pred_df['预测得分'].min():.4f}")
    lines.append(f"  平均: {pred_df['预测得分'].mean():.4f}")

    return "\n".join(lines)


def plot_forecast(scenic_name, daily_data, pred_df, output_path):
    """绘制预测趋势图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    hist = daily_data.tail(60)
    ax.plot(hist["ds"], hist["y"], "b-", label="历史情感得分", linewidth=1.5, alpha=0.7)

    ax.plot(pred_df["ds"], pred_df["预测得分"], "r-", label="预测得分", linewidth=2)
    ax.fill_between(
        pred_df["ds"],
        pred_df["下限"],
        pred_df["上限"],
        color="red", alpha=0.15, label="预测区间"
    )

    ax.axhline(y=WARNING_THRESHOLD, color="orange", linestyle="--", label=f"预警阈值({WARNING_THRESHOLD})")

    wdf = pred_df[pred_df["预警"]]
    if len(wdf) > 0:
        ax.scatter(wdf["ds"], wdf["预测得分"], color="red", s=50, zorder=5, label="预警日")

    ax.set_xlabel("日期")
    ax.set_ylabel("情感得分")
    ax.set_title(f"{scenic_name} 情感得分预测趋势")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """主函数：批量处理所有景区"""
    print("=" * 50)
    print("预测预警程序 (批量版)")
    print("=" * 50)

    if not os.path.exists(INPUT_DIR):
        print(f"错误: {INPUT_DIR} 目录不存在")
        return

    files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
    if not files:
        print(f"在 {INPUT_DIR} 中未找到匹配 {FILE_PATTERN} 的文件")
        return

    print(f"找到 {len(files)} 个文件")
    print("-" * 40)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_count = 0

    for file_path in files:
        basename = os.path.basename(file_path)
        scenic_name = basename.split("_")[0]

        try:
            result = process_scenic(file_path)
            if result is None:
                continue

            scenic_name, daily_data, pred_df = result

            # 生成报告
            report = generate_forecast_report(scenic_name, daily_data, pred_df)
            report_file = os.path.join(OUTPUT_DIR, f"forecast_report_{scenic_name}_{timestamp}.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"  报告已保存: {report_file}")

            # 绘制图表
            chart_file = os.path.join(OUTPUT_DIR, f"forecast_chart_{scenic_name}_{timestamp}.png")
            plot_forecast(scenic_name, daily_data, pred_df, chart_file)
            print(f"  图表已保存: {chart_file}")

            # 保存预测数据
            pred_df_copy = pred_df.copy()
            pred_df_copy["ds"] = pred_df_copy["ds"].dt.strftime("%Y-%m-%d")
            pred_df_copy.columns = ["日期", "预测得分", "下限", "上限", "预警"]
            csv_file = os.path.join(OUTPUT_DIR, f"forecast_data_{scenic_name}_{timestamp}.csv")
            pred_df_copy.to_csv(csv_file, index=False, encoding="utf-8-sig")
            print(f"  数据已保存: {csv_file}")

            processed_count += 1

        except Exception as e:
            print(f"  处理 {scenic_name} 时出错: {e}")
            continue

    print("\n" + "-" * 40)
    print(f"处理完成，共处理 {processed_count} 个景区")
    print("=" * 50)


if __name__ == "__main__":
    main()