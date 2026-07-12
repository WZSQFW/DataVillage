import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
from collections import Counter
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="乡村文旅数据智能分析平台",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 全局CSS样式 ====================

st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background-color: #f5f7f6;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    /* 顶部标题 */
    .app-header {
        background: linear-gradient(135deg, #1a4a3a 0%, #2a6a5a 100%);
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .app-title {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .app-subtitle {
        color: rgba(255,255,255,0.75);
        font-size: 0.85rem;
    }
    .app-scenic {
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 1rem;
        border-radius: 20px;
        color: #fff;
        font-size: 0.9rem;
    }

    /* 指标卡片 */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .metric-item {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.6rem 0.4rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e6edeb;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a3c34;
        line-height: 1.3;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #7a8a8a;
        margin-top: 0.1rem;
    }
    .metric-delta {
        font-size: 0.65rem;
        font-weight: 600;
        margin-top: 0.05rem;
    }
    .delta-up { color: #2a8a6a; }
    .delta-down { color: #c94a4a; }

    /* 卡片容器 */
    .card {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.8rem 1.2rem 1rem 1.2rem;
        border: 1px solid #e6edeb;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2a4a3a;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-sub {
        font-size: 0.75rem;
        color: #8a9a9a;
        margin-bottom: 0.4rem;
    }

    /* 异常卡片 */
    .anomaly-item {
        background: #fdf6f5;
        border-left: 3px solid #d94f4f;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
    }
    .anomaly-item.good {
        background: #f5faf8;
        border-left-color: #2a8a6a;
    }
    .anomaly-date {
        font-weight: 600;
        font-size: 0.92rem;
        color: #1a3c34;
    }
    .anomaly-score {
        font-weight: 600;
        color: #d94f4f;
    }
    .anomaly-drop {
        color: #8a6a5a;
        font-size: 0.8rem;
    }
    .tag {
        background: #eef3f1;
        padding: 0.1rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        color: #3a5a4a;
        display: inline-block;
        margin: 0.1rem 0.15rem;
    }
    .suggestion {
        background: #eaf5f0;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        font-size: 0.78rem;
        color: #1a4a3a;
        margin: 0.1rem 0;
        display: inline-block;
    }

    /* 预测统计 */
    .pred-stats {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .pred-item {
        background: #f5f8f7;
        border-radius: 6px;
        padding: 0.3rem 0.2rem;
        text-align: center;
    }
    .pred-value {
        font-size: 1.2rem;
        font-weight: 700;
    }
    .pred-label {
        font-size: 0.6rem;
        color: #7a8a8a;
    }

    /* 导航栏 */
    .css-1d391kg {
        background-color: #f0f3f2;
    }
    .sidebar .sidebar-content {
        background-color: #f0f3f2;
    }

    /* 分割线 */
    hr {
        margin: 0.6rem 0;
        border: 0;
        border-top: 1px solid #e6edeb;
    }

    /* 小标签 */
    .badge-up {
        background: #e6f5ed;
        color: #1a7a4a;
        padding: 0.05rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-down {
        background: #fdeaea;
        color: #b33a3a;
        padding: 0.05rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-stable {
        background: #eef3f1;
        color: #4a6a5a;
        padding: 0.05rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-warn {
        background: #fdf0e0;
        color: #b3703a;
        padding: 0.05rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* 响应式 */
    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(3, 1fr); }
        .pred-stats { grid-template-columns: repeat(3, 1fr); }
        .app-title { font-size: 1.2rem; }
    }
</style>
""", unsafe_allow_html=True)


# ==================== 辅助函数 ====================

@st.cache_data
def find_latest_file(directory, pattern):
    if not os.path.exists(directory):
        return None
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


@st.cache_data
def load_analysis_result(scenic_name):
    data_dir = "数据分析"
    report_dir = "输出报告"

    dim_file = find_latest_file(data_dir, f"{scenic_name}_comments_with_dimensions_*.csv")
    if dim_file is None:
        return False, None, None, None, None

    df_dim = pd.read_csv(dim_file, encoding="utf-8-sig")
    df_dim["date"] = pd.to_datetime(df_dim["date"]).dt.date

    anomaly_file = find_latest_file(report_dir, f"anomaly_report_*.txt")

    forecast_file = find_latest_file(report_dir, f"forecast_data_*.csv")
    forecast_df = pd.read_csv(forecast_file, encoding="utf-8-sig") if forecast_file else None

    compare_file = find_latest_file(report_dir, f"competitor_comparison_*.csv")
    compare_df = pd.read_csv(compare_file, encoding="utf-8-sig") if compare_file else None

    return True, df_dim, anomaly_file, forecast_df, compare_df


@st.cache_data
def load_competitor_list():
    data_dir = "数据分析"
    if not os.path.exists(data_dir):
        return []
    files = glob.glob(os.path.join(data_dir, "*_comments_with_dimensions_*.csv"))
    names = []
    for f in files:
        name = os.path.basename(f).split("_")[0]
        if name and name not in names:
            names.append(name)
    return names


def generate_radar(compare_df):
    if compare_df is None or len(compare_df) < 2:
        return None
    dims = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
    labels = ["风景", "服务", "餐饮", "住宿", "交通"]
    fig = go.Figure()
    colors = ['#2a7a5a', '#4a8a7a', '#6a9a8a', '#3a6a8a', '#8a7a5a']
    for i, (_, row) in enumerate(compare_df.iterrows()):
        values = [row.get(d, 0) for d in dims]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels,
            name=row.get("景区", "未知"),
            fill='toself',
            fillcolor=f'rgba({60 + i * 20}, {140 + i * 10}, {100 + i * 5}, 0.15)',
            line=dict(width=1.5)
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10, weight=600))
        ),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        height=340,
        margin=dict(l=40, r=40, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def generate_wordcloud(text):
    if not text or len(text.strip()) < 20:
        return None
    words = jieba.cut(text)
    stopwords = {
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
        "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她",
        "们", "但", "而", "与", "或", "又", "更", "多", "少", "太", "真", "实", "想", "能", "来", "还", "只",
        "等", "从", "时", "里", "中", "大", "小", "高", "低", "老", "新", "旧", "长", "短", "远", "近", "觉得",
        "真的", "比较", "非常", "特别", "实在", "已经", "还是", "可以", "应该", "因为", "所以", "但是"
    }
    words = [w for w in words if w not in stopwords and len(w) >= 2]
    if len(words) < 5:
        return None
    freq = Counter(words)
    top = dict(freq.most_common(70))

    font_paths = ["simhei.ttf", "msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]
    fp = None
    for p in font_paths:
        if os.path.exists(p):
            fp = p
            break
    try:
        wc = WordCloud(
            font_path=fp, width=500, height=300,
            background_color='#f8fafa', max_words=70,
            relative_scaling=0.5, colormap='Greens',
            random_state=42, prefer_horizontal=0.7
        )
        wc.generate_from_frequencies(top)
        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except:
        return None


def parse_anomalies(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    import re
    pattern = r'\[异常日期\] (\d{4}-\d{2}-\d{2})\n(.*?)(?=\[异常日期\]|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for date_str, content in matches:
        item = {"date": date_str}
        s = re.search(r'当日情感得分: ([\d.]+)', content)
        item["score"] = float(s.group(1)) if s else None
        d = re.search(r'降幅: ([\d.]+)', content)
        item["drop"] = float(d.group(1)) if d else None
        kw = re.search(r'高频关键词: (.+?)(?:\n|$)', content)
        if kw:
            kw_text = kw.group(1)
            parts = re.findall(r'([^\(]+)\((\d+)\)', kw_text)
            item["keywords"] = [{"w": w.strip(), "c": int(c)} for w, c in parts]
        else:
            item["keywords"] = []
        dim = re.search(r'问题维度分析:(.*?)(?:\n\s*\n|$)', content, re.DOTALL)
        if dim:
            dims = re.findall(r'(\w+): 当日([\d.]+) \(全局([\d.]+), 差距([\d.]+)\)', dim.group(1))
            item["dimensions"] = [{"name": d, "day": float(s), "global": float(g)} for d, s, g, _ in dims]
        else:
            item["dimensions"] = []
        sug = re.search(r'改进建议:(.*?)(?:\n-{40}|\n\[|$)', content, re.DOTALL)
        if sug:
            sug_items = re.findall(r'\d+\.\s*(.+?)(?:\n|$)', sug.group(1))
            item["suggestions"] = [s.strip() for s in sug_items if s.strip()]
        else:
            item["suggestions"] = []
        results.append(item)
    return results


# ==================== 侧边栏 ====================

with st.sidebar:
    st.markdown("### 🧭 导航")

    names = load_competitor_list()
    scenic = st.selectbox("选择景区", names) if names else st.text_input("景区名称", "桃花源")

    st.markdown("---")

    page = st.radio(
        "功能",
        ["📊 数据总览", "🔍 情感分析", "⚠️ 异常归因", "📈 预测预警", "🏆 竞争对比"],
        index=0
    )

    st.markdown("---")
    st.caption("📌 数据来源：携程网")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d')}")

# ==================== 加载数据 ====================

has, df, anom_file, forecast_df, compare_df = load_analysis_result(scenic)

if not has:
    st.warning(f"未找到「{scenic}」的数据")
    st.stop()

# ==================== 顶部栏 ====================

total = len(df)
pos = (df["情感类别"] == "正面").sum() / total * 100
neg = (df["情感类别"] == "负面").sum() / total * 100
avg_s = df["score"].mean()
avg_senti = df["情感得分"].mean()

st.markdown(f"""
<div class="app-header">
    <div>
        <span class="app-title">🏞️ 乡村文旅数据智能分析</span>
        <span class="app-subtitle">  ·  多维度服务质量诊断</span>
    </div>
    <div class="app-scenic">📍 {scenic}</div>
</div>
""", unsafe_allow_html=True)

# 指标卡片
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-item">
        <div class="metric-value">{total}</div>
        <div class="metric-label">评论总数</div>
    </div>
    <div class="metric-item">
        <div class="metric-value">{avg_s:.2f}</div>
        <div class="metric-label">平均评分</div>
    </div>
    <div class="metric-item">
        <div class="metric-value">{avg_senti:.3f}</div>
        <div class="metric-label">情感得分</div>
    </div>
    <div class="metric-item">
        <div class="metric-value" style="color:#2a8a6a;">{pos:.1f}%</div>
        <div class="metric-label">正面占比</div>
    </div>
    <div class="metric-item">
        <div class="metric-value" style="color:#c94a4a;">{neg:.1f}%</div>
        <div class="metric-label">负面占比</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== 页面内容 ====================

# ---- 数据总览 ----
if page == "📊 数据总览":
    st.markdown("### 📈 情感变化趋势")

    daily = df.groupby("date").agg({"情感得分": "mean"}).reset_index()
    fig = px.line(daily, x="date", y="情感得分",
                  labels={"date": "", "情感得分": ""},
                  color_discrete_sequence=['#2a7a5a'])
    fig.add_hline(y=0.45, line_dash="dash", line_color="#d94f4f",
                  annotation_text="预警线 0.45", annotation_position="top right")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(showgrid=True, gridcolor='#eef3f1'),
                      yaxis=dict(showgrid=True, gridcolor='#eef3f1', range=[0, 1]))
    fig.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 评分分布")
        fig = px.histogram(df, x="score", nbins=20,
                           labels={"score": "", "count": ""},
                           color_discrete_sequence=['#3a8a7a'])
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=20),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          bargap=0.08, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 情感分类")
        cnt = df["情感类别"].value_counts().reset_index()
        cnt.columns = ["类别", "数量"]
        fig = px.pie(cnt, values="数量", names="类别",
                     color="类别",
                     color_discrete_map={"正面": "#2d8f6f", "中性": "#d4a84a", "负面": "#d94f4f"},
                     hole=0.4)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)


# ---- 情感分析 ----
elif page == "🔍 情感分析":
    st.markdown("### 🎯 五维度质量评估")
    st.caption("各维度反映评论提及该维度的频率与正向程度")

    dims = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
    labels = ["风景", "服务", "餐饮", "住宿", "交通"]
    vals = [df[d].mean() for d in dims]
    colors = ['#1a7a5a', '#2a8a6a', '#4a9a7a', '#6aaaaa', '#8abaaa']

    fig = px.bar(x=labels, y=vals, color=labels, color_discrete_sequence=colors,
                 labels={"x": "", "y": "得分"}, text_auto='.3f')
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=20),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(range=[0, 0.8], showgrid=True, gridcolor='#eef3f1'),
                      showlegend=False)
    fig.update_traces(textposition='outside', textfont=dict(size=12, weight=600))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### ☁️ 评论词云")
    txt = " ".join(df["content_cleaned"].astype(str))
    wc = generate_wordcloud(txt)
    if wc:
        st.pyplot(wc)
    else:
        st.info("词云生成失败")

    with st.expander("📋 查看评论列表"):
        st.dataframe(df[["date", "userName", "content_cleaned", "score", "情感类别"]].head(50),
                     use_container_width=True, hide_index=True)


# ---- 异常归因 ----
elif page == "⚠️ 异常归因":
    st.markdown("### ⚠️ 异常波动检测")
    st.caption("情感得分显著低于7日移动平均的日期被标记为异常")

    anomalies = parse_anomalies(anom_file)

    if anomalies is None:
        st.info("未找到异常报告")
    elif len(anomalies) == 0:
        st.success("✅ 未发现异常波动，服务质量稳定")
    else:
        for item in sorted(anomalies, key=lambda x: x.get("drop", 0), reverse=True)[:10]:
            date = item["date"]
            score = item.get("score", 0)
            drop = item.get("drop", 0)
            kw = item.get("keywords", [])
            dims = item.get("dimensions", [])
            sug = item.get("suggestions", [])

            st.markdown(f"""
            <div class="anomaly-item">
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                    <span class="anomaly-date">{date}</span>
                    <span>
                        <span class="anomaly-score">{score:.3f}</span>
                        <span class="anomaly-drop"> 降幅 {drop:.3f}</span>
                    </span>
                </div>
            """, unsafe_allow_html=True)

            if kw:
                tags = " ".join([f'<span class="tag">{k["w"]}({k["c"]})</span>' for k in kw[:6]])
                st.markdown(f'<div style="margin:0.25rem 0;">{tags}</div>', unsafe_allow_html=True)

            if dims:
                txt = "、".join([f'{d["name"]}(当日{d["day"]:.2f})' for d in dims[:3]])
                st.markdown(f'<div style="font-size:0.8rem; color:#5a6a6a;">问题维度：{txt}</div>',
                            unsafe_allow_html=True)

            if sug:
                shtml = "".join([f'<span class="suggestion">• {s}</span> ' for s in sug[:3]])
                st.markdown(f'<div style="margin-top:0.2rem;">{shtml}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ---- 预测预警 ----
elif page == "📈 预测预警":
    st.markdown("### 📈 未来30天预测")
    st.caption("基于历史趋势预测情感得分变化，橙色虚线为预警阈值")

    if forecast_df is not None:
        forecast_df["日期"] = pd.to_datetime(forecast_df["日期"])

        avgp = forecast_df["预测得分"].mean()
        minp = forecast_df["预测得分"].min()
        maxp = forecast_df["预测得分"].max()
        wcnt = forecast_df["预警"].sum()

        first = forecast_df["预测得分"].iloc[0]
        last = forecast_df["预测得分"].iloc[-1]
        if last - first > 0.04:
            trend = "上升 📈"
            tcls = "badge-up"
        elif last - first < -0.04:
            trend = "下降 📉"
            tcls = "badge-down"
        else:
            trend = "平稳 ➡️"
            tcls = "badge-stable"

        st.markdown(f"""
        <div class="pred-stats">
            <div class="pred-item"><div class="pred-value">{avgp:.3f}</div><div class="pred-label">预测均值</div></div>
            <div class="pred-item"><div class="pred-value" style="color:#1a7a5a;">{maxp:.3f}</div><div class="pred-label">最高点</div></div>
            <div class="pred-item"><div class="pred-value" style="color:#c94a4a;">{minp:.3f}</div><div class="pred-label">最低点</div></div>
            <div class="pred-item"><div class="pred-value" style="color:{'#c94a4a' if wcnt > 0 else '#2a8a6a'};">{wcnt}天</div><div class="pred-label">预警天数</div></div>
            <div class="pred-item"><div><span class="{tcls}">{trend}</span></div><div class="pred-label">趋势方向</div></div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["日期"], y=forecast_df["预测得分"],
            mode="lines+markers", name="预测",
            line=dict(color="#d94f4f", width=2.5),
            marker=dict(size=5, color="#d94f4f")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["日期"], y=forecast_df["上限"],
            mode="lines", name="置信区间",
            line=dict(color="#d94f4f", width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["日期"], y=forecast_df["下限"],
            mode="lines", name="置信区间",
            line=dict(color="#d94f4f", width=0),
            fill="tonexty", fillcolor="rgba(217,79,79,0.12)"
        ))
        fig.add_hline(y=0.45, line_dash="dash", line_color="#e8a030",
                      annotation_text="阈值 0.45", annotation_position="top right")

        wdf = forecast_df[forecast_df["预警"]]
        if len(wdf) > 0:
            fig.add_trace(go.Scatter(
                x=wdf["日期"], y=wdf["预测得分"],
                mode="markers", name="预警日",
                marker=dict(size=10, color="#d94f4f", symbol="x")
            ))

        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=20),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(showgrid=True, gridcolor='#eef3f1', tickformat='%m-%d'),
                          yaxis=dict(showgrid=True, gridcolor='#eef3f1', range=[0.3, 0.95]),
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

        if wcnt > 0:
            days = forecast_df[forecast_df["预警"]]["日期"].dt.strftime('%Y-%m-%d').tolist()
            st.warning(f"⚠️ {wcnt} 个预警日：{', '.join(days)}")
        else:
            st.success("✅ 未来30天无预警")

        with st.expander("📊 查看预测数据"):
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    else:
        st.info("未找到预测数据")


# ---- 竞争对比 ----
else:
    st.markdown("### 🏆 竞品对比分析")
    st.caption("多维度对比，识别优势与短板")

    if compare_df is not None and len(compare_df) > 1:
        radar = generate_radar(compare_df)
        if radar:
            st.plotly_chart(radar, use_container_width=True)

        cols = ["景区", "评论数", "情感得分均值", "评分均值", "正面占比", "负面占比"]
        avail = [c for c in cols if c in compare_df.columns]
        st.dataframe(compare_df[avail], use_container_width=True, hide_index=True)

        st.markdown("#### 排名")
        rank = compare_df.sort_values("情感得分均值", ascending=False)
        rank["排名"] = range(1, len(rank) + 1)
        st.dataframe(rank[["排名", "景区", "情感得分均值", "正面占比"]], use_container_width=True, hide_index=True)
    else:
        st.info("未找到对比数据")

# ==================== 页脚 ====================

st.markdown("---")
st.caption("🏞️ 乡村文旅数据智能分析平台  ·  数据来源：携程网公开评论  ·  仅供学习研究")