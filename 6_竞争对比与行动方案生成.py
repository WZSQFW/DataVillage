import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from collections import Counter
import json
import jieba

# ==================== 配置区 ====================

INPUT_DIR = "数据分析"
OUTPUT_DIR = "输出报告"
FILE_PATTERN = "*_comments_with_dimensions_*.csv"

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


def calculate_metrics(df, label):
    """计算景区核心指标"""
    metrics = {
        "景区": label,
        "评论数": len(df),
        "情感得分均值": df["情感得分"].mean(),
        "评分均值": df["score"].mean() if "score" in df.columns else 0,
        "正面占比": (df["情感类别"] == "正面").sum() / len(df) * 100,
        "负面占比": (df["情感类别"] == "负面").sum() / len(df) * 100,
        "风景_得分": df["风景_得分"].mean(),
        "服务_得分": df["服务_得分"].mean(),
        "餐饮_得分": df["餐饮_得分"].mean(),
        "住宿_得分": df["住宿_得分"].mean(),
        "交通_得分": df["交通_得分"].mean()
    }
    return metrics


def find_weakest_dimension(main_metrics, competitor_metrics):
    """找出与竞品差距最大的维度"""
    dims = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
    dim_names = ["风景", "服务", "餐饮", "住宿", "交通"]

    max_gap = 0
    weakest_dim = None
    weakest_competitor = None

    for comp in competitor_metrics:
        for dim, dim_name in zip(dims, dim_names):
            gap = comp[dim] - main_metrics[dim]
            if gap > max_gap:
                max_gap = gap
                weakest_dim = dim_name
                weakest_competitor = comp["景区"]

    return weakest_dim, max_gap, weakest_competitor


def generate_action_plan(main_metrics, competitor_metrics, top_issues):
    """生成行动方案"""
    lines = []
    lines.append(f"# {main_metrics['景区']} 服务质量提升行动方案")
    lines.append("")
    lines.append(f"**生成日期**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    lines.append("## 一、现状诊断")
    lines.append("")
    lines.append("### 1.1 核心指标")
    lines.append("| 指标 | 本景区 | 竞品均值 | 对比 |")
    lines.append("|------|--------|----------|------|")

    comp_avg = {}
    for key in ["情感得分均值", "评分均值", "正面占比", "负面占比"]:
        comp_avg[key] = np.mean([c[key] for c in competitor_metrics]) if competitor_metrics else 0

    comp_names = ", ".join([c["景区"] for c in competitor_metrics]) if competitor_metrics else "暂无竞品"

    lines.append(
        f"| 情感得分 | {main_metrics['情感得分均值']:.3f} | {comp_avg['情感得分均值']:.3f} | {'偏低' if main_metrics['情感得分均值'] < comp_avg['情感得分均值'] else '持平或偏高'} |")
    lines.append(
        f"| 平均评分 | {main_metrics['评分均值']:.2f} | {comp_avg['评分均值']:.2f} | {'偏低' if main_metrics['评分均值'] < comp_avg['评分均值'] else '持平或偏高'} |")
    lines.append(
        f"| 正面占比 | {main_metrics['正面占比']:.1f}% | {comp_avg['正面占比']:.1f}% | {'偏低' if main_metrics['正面占比'] < comp_avg['正面占比'] else '持平或偏高'} |")
    lines.append(
        f"| 负面占比 | {main_metrics['负面占比']:.1f}% | {comp_avg['负面占比']:.1f}% | {'偏高' if main_metrics['负面占比'] > comp_avg['负面占比'] else '持平或偏低'} |")
    lines.append("")

    if competitor_metrics:
        lines.append("### 1.2 各维度得分对比")
        lines.append("| 维度 | 本景区 | 竞品均值 | 差距 |")
        lines.append("|------|--------|----------|------|")

        dim_names = ["风景", "服务", "餐饮", "住宿", "交通"]
        dim_keys = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]

        for dim_name, dim_key in zip(dim_names, dim_keys):
            main_val = main_metrics[dim_key]
            comp_val = np.mean([c[dim_key] for c in competitor_metrics])
            gap = comp_val - main_val
            gap_str = f"+{gap:.3f}" if gap > 0 else f"{gap:.3f}"
            lines.append(f"| {dim_name} | {main_val:.3f} | {comp_val:.3f} | {gap_str} |")
        lines.append("")

        weakest_dim, max_gap, weakest_comp = find_weakest_dimension(main_metrics, competitor_metrics)
        lines.append(f"**最大短板**: {weakest_dim}（比{weakest_comp}低{max_gap:.3f}）")
        lines.append("")

    lines.append("### 1.3 高频负面问题")
    lines.append("")
    if top_issues:
        lines.append("| 问题关键词 | 出现频次 |")
        lines.append("|------------|----------|")
        for word, count in top_issues[:8]:
            lines.append(f"| {word} | {count} |")
    else:
        lines.append("暂无明显负面问题")
    lines.append("")

    lines.append("## 二、改进建议")
    lines.append("")

    if competitor_metrics:
        dim_actions = {
            "风景": ["优化景区游览路线设计", "增加观景平台", "加强植被养护"],
            "服务": ["开展服务礼仪专项培训", "设置投诉快速响应通道", "增加导游讲解"],
            "餐饮": ["丰富餐饮品类", "加强卫生管理", "合理定价公示菜单"],
            "住宿": ["提升设施品质", "加强清洁管理", "完善预订体验"],
            "交通": ["优化交通动线", "增加停车位供给", "实施分流管制"]
        }
        weakest_dim, _, _ = find_weakest_dimension(main_metrics, competitor_metrics)
        actions = dim_actions.get(weakest_dim, ["建立游客反馈机制", "定期分析改进"])
        for i, action in enumerate(actions, 1):
            lines.append(f"### 2.{i} {action}")
            lines.append("")
    else:
        lines.append("### 2.1 建立游客满意度月度监测机制")
        lines.append("### 2.2 定期分析负面评论并制定改进措施")
        lines.append("### 2.3 加强景区基础设施日常维护")

    lines.append("## 三、预期效果")
    lines.append("")
    lines.append("若以上建议得到有效实施，预计可达成：")
    lines.append("")
    lines.append("1. **游客满意度提升**: 情感得分预计提升 5%-15%")
    lines.append("2. **负面评论减少**: 负面占比预计下降 20%-30%")
    lines.append("3. **经济收益增长**: 二次消费收入预计增长 15%-25%")
    lines.append("")

    lines.append("---")
    lines.append(f"*本报告自动生成，数据来源为携程网公开评论，分析周期截至{datetime.now().strftime('%Y年%m月%d日')}*")

    return "\n".join(lines)


def main():
    """主函数：批量生成所有景区的行动方案"""
    print("=" * 50)
    print("竞争对比与行动方案生成程序 (批量版)")
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

    # 加载所有景区数据
    all_metrics = []
    all_dfs = {}

    for file_path in files:
        basename = os.path.basename(file_path)
        scenic_name = basename.split("_")[0]

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            df["date"] = pd.to_datetime(df["date"]).dt.date
            all_dfs[scenic_name] = df
            metrics = calculate_metrics(df, scenic_name)
            all_metrics.append(metrics)
            print(f"加载: {scenic_name} ({len(df)} 条)")
        except Exception as e:
            print(f"加载 {basename} 失败: {e}")

    if len(all_dfs) == 0:
        print("未加载到任何数据")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 为每个景区生成行动方案
    for scenic_name, df in all_dfs.items():
        print(f"\n生成行动方案: {scenic_name}")

        main_metrics = calculate_metrics(df, scenic_name)
        competitor_metrics = [m for m in all_metrics if m["景区"] != scenic_name]

        # 提取负面关键词
        negative_df = df[df["情感类别"] == "负面"]
        if len(negative_df) > 0:
            negative_text = " ".join(negative_df["content_cleaned"].astype(str))
            top_issues = extract_keywords(negative_text, 10)
        else:
            top_issues = []

        plan_text = generate_action_plan(main_metrics, competitor_metrics, top_issues)

        # 保存
        md_file = os.path.join(OUTPUT_DIR, f"action_plan_{scenic_name}_{timestamp}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(plan_text)
        print(f"  已保存: {md_file}")

        txt_file = os.path.join(OUTPUT_DIR, f"action_plan_{scenic_name}_{timestamp}.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(plan_text)
        print(f"  已保存: {txt_file}")

    # 生成汇总对比表
    compare_df = pd.DataFrame(all_metrics)
    compare_file = os.path.join(OUTPUT_DIR, f"competitor_comparison_{timestamp}.csv")
    compare_df.to_csv(compare_file, index=False, encoding="utf-8-sig")
    print(f"\n汇总对比表已保存: {compare_file}")

    print("\n" + "-" * 40)
    print(f"处理完成，共处理 {len(all_dfs)} 个景区")
    print("=" * 50)


if __name__ == "__main__":
    main()