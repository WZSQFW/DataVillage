import pandas as pd
import re
import os
import glob
from datetime import datetime

# ==================== 配置区 ====================

INPUT_DIR = "数据采集"
OUTPUT_DIR = "数据预处理"
FILE_PATTERN = "*_comments_*.csv"  # 匹配所有景区评论文件

# 日期解析模式
DATE_PATTERNS = [
    r'(\d{13})',  # 13位毫秒时间戳
    r'/Date\((\d+)[+-]\d{4}\)/',
]


# ==================== 核心函数 ====================

def clean_text(text):
    """清洗评论文本"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、：；""''（）\s]', '', text)
    return text.strip()


def parse_publish_time(time_val):
    """
    解析评论时间戳，支持多种格式
    """
    if time_val is None:
        return None

    # 数值类型（整数或浮点数）
    if isinstance(time_val, (int, float)):
        try:
            timestamp_ms = int(time_val)
            timestamp_s = timestamp_ms / 1000
            return datetime.fromtimestamp(timestamp_s)
        except:
            return None

    # 字符串类型
    if not isinstance(time_val, str):
        return None

    time_str = time_val.strip()
    if time_str == '':
        return None

    # 尝试直接解析为数字（毫秒时间戳）
    try:
        timestamp_ms = int(time_str)
        timestamp_s = timestamp_ms / 1000
        return datetime.fromtimestamp(timestamp_s)
    except ValueError:
        pass

    # 尝试正则提取数字
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, time_str)
        if match:
            try:
                timestamp_ms = int(match.group(1))
                timestamp_s = timestamp_ms / 1000
                return datetime.fromtimestamp(timestamp_s)
            except:
                continue

    return None


def get_season(month):
    if month in [3, 4, 5]:
        return "春季"
    elif month in [6, 7, 8]:
        return "夏季"
    elif month in [9, 10, 11]:
        return "秋季"
    else:
        return "冬季"


def preprocess_data(df, scenic_name):
    """
    执行数据清洗与特征工程
    """
    df_clean = df.copy()

    print(f"  原始条数: {len(df_clean)}")

    # 1. 删除重复
    df_clean = df_clean.drop_duplicates()
    print(f"  去重后: {len(df_clean)} 条")

    # 2. 删除空评论
    if "content" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["content"])
        df_clean = df_clean[df_clean["content"].str.strip() != ""]
    print(f"  删除空评论: {len(df_clean)} 条")

    # 3. 清洗评论内容
    if "content" in df_clean.columns:
        df_clean["content_cleaned"] = df_clean["content"].apply(clean_text)
        df_clean = df_clean[df_clean["content_cleaned"] != ""]
    print(f"  清洗后: {len(df_clean)} 条")

    # 4. 处理评分
    if "score" in df_clean.columns:
        df_clean["score"] = pd.to_numeric(df_clean["score"], errors="coerce")
        mean_score = df_clean["score"].mean()
        df_clean["score"] = df_clean["score"].fillna(mean_score)

    # 5. 解析时间
    if "publishTime" in df_clean.columns:
        df_clean["publish_datetime"] = df_clean["publishTime"].apply(parse_publish_time)
        df_clean = df_clean.dropna(subset=["publish_datetime"])
    print(f"  时间解析后: {len(df_clean)} 条")

    if len(df_clean) == 0:
        print(f"  警告: {scenic_name} 时间解析全部失败")
        return df_clean

    # 6. 提取日期特征
    df_clean["date"] = df_clean["publish_datetime"].dt.date
    df_clean["month"] = df_clean["publish_datetime"].dt.month
    df_clean["weekday"] = df_clean["publish_datetime"].dt.weekday
    df_clean["hour"] = df_clean["publish_datetime"].dt.hour
    df_clean["season"] = df_clean["month"].apply(get_season)

    # 7. 评论长度
    df_clean["content_length"] = df_clean["content_cleaned"].str.len()

    # 8. 评分等级
    def score_level(s):
        if s <= 2:
            return "低"
        elif s <= 3:
            return "中"
        else:
            return "高"

    if "score" in df_clean.columns:
        df_clean["score_level"] = df_clean["score"].apply(score_level)

    # 9. 有用数标记
    if "usefulCount" in df_clean.columns:
        mean_useful = df_clean["usefulCount"].mean()
        df_clean["high_useful"] = df_clean["usefulCount"] > mean_useful

    print(f"  最终保留: {len(df_clean)} 条")
    return df_clean


def main():
    """主函数：批量处理所有景区"""
    print("=" * 50)
    print("数据清洗与预处理程序 (批量版)")
    print("=" * 50)

    if not os.path.exists(INPUT_DIR):
        print(f"错误: {INPUT_DIR} 目录不存在")
        return

    # 查找所有评论文件
    files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
    if not files:
        print(f"在 {INPUT_DIR} 中未找到匹配 {FILE_PATTERN} 的文件")
        return

    print(f"找到 {len(files)} 个文件")
    print("-" * 40)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    processed_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for file_path in files:
        # 提取景区名称（文件名第一个下划线之前的部分）
        basename = os.path.basename(file_path)
        scenic_name = basename.split("_")[0]

        print(f"\n处理: {scenic_name}")
        print(f"  文件: {basename}")

        try:
            df_raw = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"  读取成功: {len(df_raw)} 条")

            df_clean = preprocess_data(df_raw, scenic_name)

            if len(df_clean) == 0:
                print(f"  跳过: {scenic_name} 清洗后无数据")
                continue

            # 保存
            output_file = os.path.join(OUTPUT_DIR, f"{scenic_name}_comments_cleaned_{timestamp}.csv")
            df_clean.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"  已保存: {output_file}")
            processed_count += 1

        except Exception as e:
            print(f"  处理 {scenic_name} 时出错: {e}")
            continue

    print("\n" + "-" * 40)
    print(f"处理完成，共处理 {processed_count} 个景区")
    print("=" * 50)


if __name__ == "__main__":
    main()