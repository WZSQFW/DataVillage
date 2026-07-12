import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from collections import Counter
import jieba
import json

# ==================== 配置区 ====================

INPUT_DIR = "数据分析"
OUTPUT_DIR = "输出报告"
FILE_PATTERN = "*_comments_with_dimensions_*.csv"

ANOMALY_THRESHOLD = 1.5
MIN_DAILY_COUNT = 3
MIN_MA_WINDOW = 5

STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "它", "他", "她", "们", "但", "而", "与", "或", "又", "更",
    "多", "少", "太", "真", "实", "想", "能", "来", "还", "只", "等", "从",
    "时", "里", "中", "大", "小", "高", "低", "老", "新", "旧", "长", "短", "远", "近"
}


# ==================== 核心函数 ====================

def extract_keywords(text, top_n=10):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    words = jieba.cut(text)
    words = [w for w in words if w not in STOPWORDS and len(w) >= 2]
    counter = Counter(words)
    return counter.most_common(top_n)


def detect_anomalies(df):
    """检测情感得分的异常下降点"""
    daily = df.groupby("date").agg({
        "情感得分": "mean",
        "content_cleaned": lambda x: " ".join(x.astype(str))
    }).reset_index()

    daily["评论数"] = df.groupby("date").size().values
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["MA"] = daily["情感得分"].rolling(7, min_periods=MIN_MA_WINDOW).mean()
    daily["STD"] = daily["情感得分"].rolling(7, min_periods=MIN_MA_WINDOW).std()

    condition = (
            (daily["情感得分"] < (daily["MA"] - ANOMALY_THRESHOLD * daily["STD"])) &
            (daily["STD"].notna()) &
            (daily["评论数"] >= MIN_DAILY_COUNT)
    )

    daily["is_anomaly"] = condition
    daily["降幅"] = daily["MA"] - daily["情感得分"]
    anomaly_dates = daily[daily["is_anomaly"]].sort_values("降幅", ascending=False)

    return daily, anomaly_dates


def analyze_problem_dimensions(df, anomaly_date):
    day_data = df[df["date"] == anomaly_date]
    dims = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
    global_avg = df[dims].mean()
    day_avg = day_data[dims].mean()

    problem_dims = []
    for dim in dims:
        dim_name = dim.replace("_得分", "")
        diff = global_avg[dim] - day_avg[dim]
        if diff > 0.05:
            problem_dims.append({
                "维度": dim_name,
                "全局均值": round(global_avg[dim], 3),
                "当日均值": round(day_avg[dim], 3),
                "差值": round(diff, 3)
            })
    return sorted(problem_dims, key=lambda x: x["差值"], reverse=True)


def generate_suggestions(problem_dim, keywords):
    dim_suggestions = {
        "服务": ["加强景区工作人员服务礼仪培训", "设置游客投诉快速响应通道", "开展服务满意度月度评比"],
        "交通": ["优化景区内部交通动线设计", "增加停车位供给", "在节假日实施交通分流管制"],
        "餐饮": ["丰富餐饮品类，增加地方特色", "加强餐饮卫生管理", "合理定价公示菜单"],
        "住宿": ["提升民宿设施品质", "加强客房清洁管理", "完善线上预订体验"],
        "风景": ["优化景区游览路线", "增加观景平台", "加强植被养护和景观维护"]
    }

    suggestions = []
    if problem_dim:
        for item in problem_dim:
            dim_name = item["维度"]
            if dim_name in dim_suggestions:
                suggestions.extend(dim_suggestions[dim_name][:2])
    else:
        suggestions = ["建立游客满意度月度监测机制", "定期分析负面评论并制定改进措施"]

    keyword_suggestions = {
        "门票": "优化门票定价策略", "排队": "引入分时预约系统",
        "停车": "扩建停车场", "卫生": "增加保洁频次",
        "态度": "加强服务意识培训", "设施": "更新基础设施"
    }

    for word, _ in keywords[:3]:
        for kw, sug in keyword_suggestions.items():
            if kw in word:
                suggestions.append(sug)

    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:5]


def process_scenic(file_path):
    """处理单个景区的异常分析"""
    basename = os.path.basename(file_path)
    scenic_name = basename.split("_")[0]

    print(f"\n处理: {scenic_name}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  数据: {len(df)} 条")

    daily, anomaly_dates = detect_anomalies(df)

    if len(anomaly_dates) == 0:
        print("  无异常波动")
        return None, []

    print(f"  发现 {len(anomaly_dates)} 个异常日期")
    return df, anomaly_dates


def generate_anomaly_report(scenic_name, df, anomaly_dates):
    """生成单个景区的异常报告"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"异常归因分析报告 - {scenic_name}")
    lines.append("=" * 60)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"评论总数: {len(df)}")
    lines.append(f"异常日期数: {len(anomaly_dates)}")
    lines.append("")

    if len(anomaly_dates) == 0:
        lines.append("结论: 未发现明显异常波动，服务质量整体稳定。")
        return "\n".join(lines)

    for idx, row in anomaly_dates.head(10).iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        score = row["情感得分"]
        ma = row["MA"]
        drop = row["降幅"]
        count = row["评论数"]

        lines.append(f"\n[异常日期] {date_str}")
        lines.append(f"  当日情感得分: {score:.4f} (评论数: {count}条)")
        lines.append(f"  7日移动平均: {ma:.4f}")
        lines.append(f"  降幅: {drop:.4f}")

        day_text = row["content_cleaned"]
        keywords = extract_keywords(day_text, 8)
        if keywords:
            kw_str = ", ".join([f"{w}({c})" for w, c in keywords])
            lines.append(f"  高频关键词: {kw_str}")

        problem_dims = analyze_problem_dimensions(df, row["date"])
        if problem_dims:
            lines.append("  问题维度分析:")
            for dim in problem_dims:
                lines.append(
                    f"    {dim['维度']}: 当日{dim['当日均值']:.3f} (全局{dim['全局均值']:.3f}, 差距{dim['差值']:.3f})")

        suggestions = generate_suggestions(problem_dims, keywords)
        if suggestions:
            lines.append("  改进建议:")
            for i, s in enumerate(suggestions, 1):
                lines.append(f"    {i}. {s}")

        lines.append("-" * 40)

    return "\n".join(lines)


def main():
    """主函数：批量处理所有景区"""
    print("=" * 50)
    print("异常归因分析程序 (批量版)")
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
                # 无异常，生成简短报告
                report = f"{scenic_name}: 无异常波动，服务质量稳定"
                report_file = os.path.join(OUTPUT_DIR, f"anomaly_report_{scenic_name}_{timestamp}.txt")
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write(report)
                processed_count += 1
                continue

            df, anomaly_dates = result
            report = generate_anomaly_report(scenic_name, df, anomaly_dates)

            report_file = os.path.join(OUTPUT_DIR, f"anomaly_report_{scenic_name}_{timestamp}.txt")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"  报告已保存: {report_file}")
            processed_count += 1

        except Exception as e:
            print(f"  处理 {scenic_name} 时出错: {e}")
            continue

    print("\n" + "-" * 40)
    print(f"处理完成，共处理 {processed_count} 个景区")
    print("=" * 50)


if __name__ == "__main__":
    main()