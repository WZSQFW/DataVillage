import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from snownlp import SnowNLP
import jieba

# ==================== 配置区 ====================

INPUT_DIR = "数据预处理"
OUTPUT_DIR = "数据分析"
FILE_PATTERN = "*_comments_cleaned_*.csv"

# 五维度关键词
DIMENSION_KEYWORDS = {
    "风景": ["风景", "景色", "山水", "自然", "美", "漂亮", "壮观", "秀丽", "桃花", "花", "湖", "山", "景", "环境",
             "绿化", "视野"],
    "服务": ["服务", "态度", "热情", "周到", "耐心", "专业", "冷漠", "差劲", "工作人员", "导游", "讲解", "服务台",
             "接待"],
    "餐饮": ["吃", "饭", "菜", "美食", "口味", "味道", "好吃", "难吃", "餐厅", "小吃", "特色", "农家乐", "饮品",
             "价格"],
    "住宿": ["住", "房间", "民宿", "酒店", "床", "干净", "卫生", "舒适", "客栈", "住得", "环境", "隔音", "设施"],
    "交通": ["交通", "路", "停车", "车", "堵", "方便", "到达", "位置", "公交", "自驾", "门票", "摆渡", "观光车"]
}


# ==================== 核心函数 ====================

def get_sentiment_score(text):
    """使用SnowNLP计算情感得分"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.5
    try:
        if len(text) > 300:
            text = text[:300]
        return SnowNLP(text).sentiments
    except Exception:
        return 0.5


def classify_sentiment(score):
    """情感分类"""
    if score >= 0.6:
        return "正面"
    elif score >= 0.4:
        return "中性"
    else:
        return "负面"


def dimension_score(text, keywords):
    """计算单维度得分"""
    if not isinstance(text, str):
        return 0.0
    text = text.lower()
    match_count = sum(1 for kw in keywords if kw in text)
    return min(match_count / 3.0, 1.0)


def analyze_dimensions(df):
    """五维度拆解"""
    df_result = df.copy()

    for dim, keywords in DIMENSION_KEYWORDS.items():
        col_name = f"{dim}_得分"
        df_result[col_name] = df_result["content_cleaned"].apply(
            lambda x: dimension_score(x, keywords)
        )

    dim_cols = [f"{dim}_得分" for dim in DIMENSION_KEYWORDS.keys()]
    df_result["综合维度分"] = df_result[dim_cols].mean(axis=1)

    def get_top_dimension(row):
        scores = {dim: row[f"{dim}_得分"] for dim in DIMENSION_KEYWORDS.keys()}
        if max(scores.values()) == 0:
            return "无明确维度"
        return max(scores, key=scores.get)

    df_result["主要维度"] = df_result.apply(get_top_dimension, axis=1)

    return df_result


def main():
    """主函数：批量处理所有景区"""
    print("=" * 50)
    print("情感分析与五维度拆解程序 (批量版)")
    print("=" * 50)

    if not os.path.exists(INPUT_DIR):
        print(f"错误: {INPUT_DIR} 目录不存在")
        return

    files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
    if not files:
        print(f"在 {INPUT_DIR} 中未找到匹配 {FILE_PATTERN} 的文件")
        print("请先运行 2_数据清洗与预处理.py")
        return

    print(f"找到 {len(files)} 个文件")
    print("-" * 40)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    processed_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_reports = []

    for file_path in files:
        basename = os.path.basename(file_path)
        scenic_name = basename.split("_")[0]

        print(f"\n处理: {scenic_name}")
        print(f"  文件: {basename}")

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"  读取成功: {len(df)} 条")

            # 确保有评论文本
            if "content_cleaned" not in df.columns:
                print(f"  跳过: {scenic_name} 缺少 content_cleaned 列")
                continue

            # 情感分析
            print("  执行情感分析...")
            df["情感得分"] = df["content_cleaned"].apply(get_sentiment_score)
            df["情感类别"] = df["情感得分"].apply(classify_sentiment)

            # 维度拆解
            print("  执行五维度拆解...")
            df = analyze_dimensions(df)

            # 保存
            output_file = os.path.join(OUTPUT_DIR, f"{scenic_name}_comments_with_dimensions_{timestamp}.csv")
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"  已保存: {output_file}")

            # 统计
            pos = (df["情感类别"] == "正面").sum()
            neu = (df["情感类别"] == "中性").sum()
            neg = (df["情感类别"] == "负面").sum()
            avg_senti = df["情感得分"].mean()

            report = f"""
  {scenic_name} 统计:
    正面: {pos} 条, 中性: {neu} 条, 负面: {neg} 条
    平均情感得分: {avg_senti:.4f}
    各维度平均分: """
            for dim in DIMENSION_KEYWORDS.keys():
                report += f"{dim} {df[f'{dim}_得分'].mean():.3f}  "
            all_reports.append(report)
            print(report)

            processed_count += 1

        except Exception as e:
            print(f"  处理 {scenic_name} 时出错: {e}")
            continue

    # 生成汇总报告
    summary_file = os.path.join(OUTPUT_DIR, f"dimension_summary_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("情感分析与维度拆解汇总报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理景区数: {processed_count}\n\n")
        for r in all_reports:
            f.write(r + "\n")

    print("\n" + "-" * 40)
    print(f"处理完成，共处理 {processed_count} 个景区")
    print(f"汇总报告: {summary_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()