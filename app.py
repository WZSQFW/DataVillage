import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from datetime import datetime, timedelta
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 全局CSS样式 ====================

st.markdown("""
<style>
    .stApp { background-color: #f5f7f6; }
    .main .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

    .app-header {
        background: linear-gradient(135deg, #1a3c34 0%, #2a6a5a 100%);
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .app-title { color: #fff; font-size: 1.5rem; font-weight: 700; letter-spacing: 1px; }
    .app-subtitle { color: rgba(255,255,255,0.7); font-size: 0.8rem; }
    .app-scenic { background: rgba(255,255,255,0.15); padding: 0.15rem 1rem; border-radius: 20px; color: #fff; font-size: 0.85rem; }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-item {
        background: #fff;
        border-radius: 8px;
        padding: 0.5rem 0.3rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e6edeb;
    }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #1a3c34; line-height: 1.3; }
    .metric-label { font-size: 0.65rem; color: #7a8a8a; margin-top: 0.05rem; }

    .card {
        background: #fff;
        border-radius: 8px;
        padding: 0.6rem 1rem 0.8rem 1rem;
        border: 1px solid #e6edeb;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 0.6rem;
    }
    .card-title { font-size: 0.9rem; font-weight: 600; color: #2a4a3a; margin-bottom: 0.3rem; }

    .anomaly-critical {
        background: #fdf0ee;
        border-left: 4px solid #d94f4f;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .anomaly-warning {
        background: #fdf6ec;
        border-left: 4px solid #e8a030;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .anomaly-info {
        background: #f0f5f8;
        border-left: 4px solid #4a8aaa;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }
    .anomaly-date { font-weight: 600; font-size: 0.9rem; color: #1a3c34; }
    .anomaly-score { font-weight: 600; color: #d94f4f; }
    .anomaly-drop { color: #8a7a6a; font-size: 0.8rem; }
    .tag {
        background: #eef3f1;
        padding: 0.05rem 0.5rem;
        border-radius: 12px;
        font-size: 0.65rem;
        color: #3a5a4a;
        display: inline-block;
        margin: 0.1rem 0.1rem;
    }
    .tag-dim { background: #e6edf5; color: #3a5a7a; }
    .suggestion { background: #eaf5f0; padding: 0.1rem 0.6rem; border-radius: 4px; font-size: 0.75rem; color: #1a4a3a; display: inline-block; margin: 0.1rem 0; }

    .pred-stats {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .pred-item {
        background: #f5f8f7;
        border-radius: 6px;
        padding: 0.25rem 0.2rem;
        text-align: center;
    }
    .pred-value { font-size: 1.1rem; font-weight: 700; }
    .pred-label { font-size: 0.55rem; color: #7a8a8a; }

    .badge-up { background: #e6f5ed; color: #1a7a4a; padding: 0.05rem 0.6rem; border-radius: 12px; font-size: 0.65rem; font-weight: 600; }
    .badge-down { background: #fdeaea; color: #b33a3a; padding: 0.05rem 0.6rem; border-radius: 12px; font-size: 0.65rem; font-weight: 600; }
    .badge-stable { background: #eef3f1; color: #4a6a5a; padding: 0.05rem 0.6rem; border-radius: 12px; font-size: 0.65rem; font-weight: 600; }

    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(3, 1fr); }
        .pred-stats { grid-template-columns: repeat(3, 1fr); }
        .app-title { font-size: 1.1rem; }
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
    df = pd.read_csv(dim_file, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    anom_file = find_latest_file(report_dir, f"anomaly_report_*.txt")
    fc_file = find_latest_file(report_dir, f"forecast_data_*.csv")
    fc = pd.read_csv(fc_file, encoding="utf-8-sig") if fc_file else None
    cmp_file = find_latest_file(report_dir, f"competitor_comparison_*.csv")
    cmp = pd.read_csv(cmp_file, encoding="utf-8-sig") if cmp_file else None
    return True, df, anom_file, fc, cmp


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


def generate_wordcloud(text):
    if not text or len(text.strip()) < 20:
        return None
    words = jieba.cut(text)
    stopwords = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
                 "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她",
                 "们", "但", "而", "与", "或", "又", "更", "多", "少", "太", "真", "实", "想", "能", "来", "还", "只",
                 "等", "从", "时", "里", "中", "大", "小", "高", "低", "老", "新", "旧", "长", "短", "远", "近"}
    words = [w for w in words if w not in stopwords and len(w) >= 2]
    if len(words) < 5:
        return None
    freq = Counter(words)
    top = dict(freq.most_common(60))
    font_paths = ["simhei.ttf", "msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    fp = next((p for p in font_paths if os.path.exists(p)), None)
    try:
        wc = WordCloud(font_path=fp, width=400, height=220, background_color='#f8fafa',
                       max_words=60, relative_scaling=0.5, colormap='Greens',
                       random_state=42, prefer_horizontal=0.7)
        wc.generate_from_frequencies(top)
        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
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
            parts = re.findall(r'([^\(]+)\((\d+)\)', kw.group(1))
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


def generate_anomaly_suggestions(item):
    dims = item.get("dimensions", [])
    kw = item.get("keywords", [])
    drop = item.get("drop", 0)

    dim_names = [d["name"] for d in dims]
    kw_words = [k["w"] for k in kw[:5]]

    suggestions = []

    dim_suggestions = {
        "服务": ["开展服务礼仪专项培训", "设置游客投诉快速响应通道", "增加导游讲解频次"],
        "交通": ["优化景区内部交通动线", "增加停车位供给", "节假日实施交通分流管制"],
        "餐饮": ["丰富餐饮品类，增加地方特色", "加强餐饮卫生管理", "合理定价公示菜单"],
        "住宿": ["提升民宿设施品质", "加强客房清洁管理", "完善线上预订体验"],
        "风景": ["优化景区游览路线", "增加观景平台", "加强植被养护和景观维护"]
    }

    for dim in dim_names:
        if dim in dim_suggestions:
            suggestions.extend(dim_suggestions[dim])

    keyword_suggestions = {
        "门票": "优化门票定价策略，增加套票优惠",
        "排队": "引入分时预约系统，减少等待时间",
        "卫生": "增加保洁人员和垃圾清运频次",
        "停车": "扩建停车场或设置临时停车区",
        "指示": "完善景区导览标识系统",
        "态度": "加强员工服务意识培训",
        "贵": "建立价格公示制度，提升性价比",
        "设施": "更新景区基础设施，定期维护",
        "拥挤": "实施限流措施，提升游览体验",
        "网络": "加强景区5G信号覆盖",
        "厕所": "增加卫生设施，加强清洁管理",
        "餐饮": "丰富菜品种类，提升餐饮品质"
    }

    for word in kw_words:
        for kw_key, sug in keyword_suggestions.items():
            if kw_key in word:
                suggestions.append(sug)

    if drop > 0.4:
        suggestions.append("启动服务质量紧急整改预案")
        suggestions.append("组织管理层专项会议分析根因")
    elif drop > 0.25:
        suggestions.append("开展服务质量专项检查")
        suggestions.append("加强当日问题复盘总结")

    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique[:4]


# ==================== 季节分析辅助函数 ====================

def get_season(month):
    if month in [3, 4, 5]:
        return "春季"
    elif month in [6, 7, 8]:
        return "夏季"
    elif month in [9, 10, 11]:
        return "秋季"
    else:
        return "冬季"


def prepare_seasonal_data(df):
    df_season = df.copy()
    df_season["month"] = pd.to_datetime(df_season["date"]).dt.month
    df_season["season"] = df_season["month"].apply(get_season)
    seasonal = df_season.groupby("season").agg({
        "情感得分": "mean",
        "score": "mean"
    }).reset_index()
    season_order = ["春季", "夏季", "秋季", "冬季"]
    seasonal["season"] = pd.Categorical(seasonal["season"], categories=season_order, ordered=True)
    seasonal = seasonal.sort_values("season").reset_index(drop=True)
    return seasonal


def prepare_user_profile(df):
    profile = {}
    if "ipLocatedName" in df.columns:
        location_counts = df["ipLocatedName"].dropna()
        location_counts = location_counts[location_counts != ""]
        if len(location_counts) > 0:
            profile["locations"] = location_counts.value_counts().head(10).reset_index()
            profile["locations"].columns = ["来源地", "评论数"]
        else:
            profile["locations"] = None
    else:
        profile["locations"] = None
    
    if "touristType" in df.columns:
        type_counts = df["touristType"].dropna()
        type_counts = type_counts[type_counts != ""]
        if len(type_counts) > 0:
            profile["types"] = type_counts.value_counts().reset_index()
            profile["types"].columns = ["游客类型", "评论数"]
        else:
            profile["types"] = None
    else:
        profile["types"] = None
    
    return profile


# ==================== 侧边栏 ====================

with st.sidebar:
    st.markdown("### 导航")
    names = load_competitor_list()
    scenic = st.selectbox("选择景区", names) if names else st.text_input("景区名称", "桃花源")
    st.markdown("---")
    page = st.radio("功能", ["数据总览", "情感分析", "异常归因", "预测预警", "用户画像", "竞争对比"], index=0)
    st.markdown("---")
    
    with st.expander("数据采集指引"):
        st.markdown("""
        如需采集新景区数据：
        1. 在携程网搜索景点，从URL获取POI ID
        2. 修改 `1_爬虫_常德景点评论采集.py` 中的 `SCENIC_SPOTS`
        3. 依次运行 1-6 号脚本
        4. 刷新本页面
        """)
    
    st.caption("数据来源：携程网")
    st.caption(datetime.now().strftime('%Y-%m-%d'))

# ==================== 加载数据 ====================

has, df, anom_file, forecast_df, compare_df = load_analysis_result(scenic)

if not has:
    st.warning(f"未找到「{scenic}」的数据")
    st.stop()

# ==================== 顶部指标栏 ====================

total = len(df)
pos = (df["情感类别"] == "正面").sum() / total * 100
neg = (df["情感类别"] == "负面").sum() / total * 100
avg_s = df["score"].mean()
avg_senti = df["情感得分"].mean()

dim_cols = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
available_dims = [c for c in dim_cols if c in df.columns]
if available_dims:
    df["综合健康度"] = df[available_dims].mean(axis=1)
    health_score = df["综合健康度"].mean()
else:
    health_score = avg_senti

st.markdown(f"""
<div class="app-header">
    <div>
        <span class="app-title">乡村文旅数据智能分析</span>
        <span class="app-subtitle">  ·  多维度服务质量诊断</span>
    </div>
    <div class="app-scenic"> {scenic}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-item"><div class="metric-value">{total}</div><div class="metric-label">评论总数</div></div>
    <div class="metric-item"><div class="metric-value">{avg_s:.2f}</div><div class="metric-label">平均评分</div></div>
    <div class="metric-item"><div class="metric-value">{avg_senti:.3f}</div><div class="metric-label">情感得分</div></div>
    <div class="metric-item"><div class="metric-value" style="color:#2a8a6a;">{pos:.1f}%</div><div class="metric-label">正面占比</div></div>
    <div class="metric-item"><div class="metric-value" style="color:#c94a4a;">{neg:.1f}%</div><div class="metric-label">负面占比</div></div>
    <div class="metric-item"><div class="metric-value" style="color:#4a6a9a;">{health_score:.3f}</div><div class="metric-label">综合健康度</div></div>
</div>
""", unsafe_allow_html=True)

# ==================== 颜色方案 ====================

color_palette = ['#2a7a5a', '#4a8aaa', '#d4a84a', '#c94a4a', '#8a6a9a', '#d47a5a', '#5a8a8a']

# ==================== 数据总览 ====================

if page == "数据总览":
    st.markdown("### 情感与评论量趋势")

    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.slider("选择日期范围", min_date, max_date, (min_date, max_date), format="YYYY-MM-DD")
    df_filtered = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

    daily = df_filtered.groupby("date").agg({
        "情感得分": "mean",
        "score": "mean"
    }).reset_index()
    daily_count = df_filtered.groupby("date").size().reset_index(name="评论数")
    daily = daily.merge(daily_count, on="date", how="left")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily["date"],
        y=daily["评论数"],
        name="评论数",
        yaxis="y2",
        marker=dict(color="rgba(74, 138, 170, 0.4)", line=dict(color="rgba(74, 138, 170, 0.8)", width=1)),
        hovertemplate="日期: %{x}<br>评论数: %{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["情感得分"],
        name="情感得分",
        mode="lines+markers",
        line=dict(color="#2a7a5a", width=2.5),
        marker=dict(size=4, color="#2a7a5a"),
        hovertemplate="日期: %{x}<br>情感得分: %{y:.3f}<extra></extra>"
    ))

    fig.add_hline(y=0.45, line_dash="dash", line_color="#d94f4f", annotation_text="预警线",
                  annotation_position="top right")

    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#eef3f1'),
        yaxis=dict(
            title="情感得分",
            showgrid=True,
            gridcolor='#eef3f1',
            range=[0, 1],
            side="left"
        ),
        yaxis2=dict(
            title="评论数",
            showgrid=False,
            side="right",
            overlaying="y"
        ),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 季节性分析")
    seasonal_data = prepare_seasonal_data(df_filtered)

    if len(seasonal_data) > 0:
        season_melted = seasonal_data.melt(
            id_vars=["season"],
            value_vars=["情感得分", "score"],
            var_name="指标",
            value_name="值"
        )
        season_melted["指标"] = season_melted["指标"].replace({
            "情感得分": "平均情感得分",
            "score": "平均评分"
        })

        fig_season = px.bar(
            season_melted,
            x="season",
            y="值",
            color="指标",
            barmode="group",
            color_discrete_map={
                "平均情感得分": "#2a7a5a",
                "平均评分": "#4a8aaa"
            },
            labels={"season": "", "值": "", "指标": ""},
            text_auto='.3f'
        )
        fig_season.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='#eef3f1'),
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
        )
        fig_season.update_traces(textposition='outside', textfont=dict(size=11, weight=600))
        st.plotly_chart(fig_season, use_container_width=True)

        best_season = seasonal_data.loc[seasonal_data["情感得分"].idxmax(), "season"]
        worst_season = seasonal_data.loc[seasonal_data["情感得分"].idxmin(), "season"]
        st.caption(f"最佳季节：{best_season}（情感得分 {seasonal_data['情感得分'].max():.3f}）  |  最需关注：{worst_season}（情感得分 {seasonal_data['情感得分'].min():.3f}）")
    else:
        st.info("暂无季节性数据")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 评分分布")
        fig = px.histogram(df_filtered, x="score", nbins=20,
                           labels={"score": "", "count": ""},
                           color_discrete_sequence=['#4a8aaa'])
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=20),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          bargap=0.08, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### 情感分类")
        cnt = df_filtered["情感类别"].value_counts().reset_index()
        cnt.columns = ["类别", "数量"]
        fig = px.pie(cnt, values="数量", names="类别",
                     color="类别",
                     color_discrete_map={"正面": "#2d8f6f", "中性": "#d4a84a", "负面": "#d94f4f"},
                     hole=0.4)
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 数据导出")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        csv_data = df_filtered.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载当前筛选数据 (CSV)",
            data=csv_data,
            file_name=f"{scenic}_数据_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_d2:
        csv_full = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载完整数据 (CSV)",
            data=csv_full,
            file_name=f"{scenic}_完整数据_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_d3:
        summary = {
            "景区": scenic,
            "总评论数": total,
            "平均评分": f"{avg_s:.2f}",
            "情感得分": f"{avg_senti:.3f}",
            "正面占比": f"{pos:.1f}%",
            "负面占比": f"{neg:.1f}%",
            "综合健康度": f"{health_score:.3f}",
            "数据日期范围": f"{min_date} ~ {max_date}",
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        summary_df = pd.DataFrame([summary])
        csv_summary = summary_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载统计摘要 (CSV)",
            data=csv_summary,
            file_name=f"{scenic}_统计摘要_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==================== 情感分析 ====================

elif page == "情感分析":
    st.markdown("### 五维度质量评估")

    dims = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
    labels = ["风景", "服务", "餐饮", "住宿", "交通"]
    vals = [df[d].mean() for d in dims]

    fig = px.bar(x=labels, y=vals,
                 color=labels,
                 color_discrete_sequence=color_palette[:5],
                 labels={"x": "", "y": "得分"}, text_auto='.3f')
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=20),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(range=[0, 0.8], showgrid=True, gridcolor='#eef3f1'),
                      showlegend=False)
    fig.update_traces(textposition='outside', textfont=dict(size=12, weight=600))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看评论词云"):
        txt = " ".join(df["content_cleaned"].astype(str))
        wc = generate_wordcloud(txt)
        if wc:
            st.pyplot(wc)
        else:
            st.info("词云生成失败")

    with st.expander("查看评论列表"):
        st.dataframe(df[["date", "userName", "content_cleaned", "score", "情感类别"]].head(50),
                     use_container_width=True, hide_index=True)


# ==================== 异常归因 ====================

elif page == "异常归因":
    st.markdown("### 异常波动检测")
    st.caption("情感得分显著低于7日移动平均的日期被标记为异常，按严重程度分级")

    anomalies = parse_anomalies(anom_file)

    if anomalies is None:
        st.info("未找到异常报告，请先运行 4_异常归因分析.py")
    elif len(anomalies) == 0:
        st.success("未发现异常波动，服务质量稳定")
    else:
        dates = [a["date"] for a in anomalies]
        date_min = min(dates)
        date_max = max(dates)
        date_range = st.slider("筛选日期范围",
                               datetime.strptime(date_min, "%Y-%m-%d").date(),
                               datetime.strptime(date_max, "%Y-%m-%d").date(),
                               (datetime.strptime(date_min, "%Y-%m-%d").date(),
                                datetime.strptime(date_max, "%Y-%m-%d").date()),
                               format="YYYY-MM-DD")

        severity_filter = st.selectbox("严重程度", ["全部", "严重 (降幅>0.35)", "中等 (降幅>0.2)", "轻微 (降幅<=0.2)"])

        filtered = []
        for item in anomalies:
            d = datetime.strptime(item["date"], "%Y-%m-%d").date()
            if not (date_range[0] <= d <= date_range[1]):
                continue
            drop = item.get("drop", 0)
            if severity_filter == "严重 (降幅>0.35)" and drop <= 0.35:
                continue
            if severity_filter == "中等 (降幅>0.2)" and drop <= 0.2:
                continue
            if severity_filter == "轻微 (降幅<=0.2)" and drop > 0.2:
                continue
            filtered.append(item)

        if len(filtered) == 0:
            st.info("当前筛选条件下无异常记录")
        else:
            total_anom = len(filtered)
            critical = sum(1 for a in filtered if a.get("drop", 0) > 0.35)
            medium = sum(1 for a in filtered if 0.2 < a.get("drop", 0) <= 0.35)
            mild = sum(1 for a in filtered if a.get("drop", 0) <= 0.2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("异常总数", total_anom)
            col2.metric("严重", critical, delta="需立即关注", delta_color="off")
            col3.metric("中等", medium)
            col4.metric("轻微", mild)

            st.markdown("---")

            issue_types = {}
            for item in filtered:
                dims = item.get("dimensions", [])
                for d in dims:
                    name = d["name"]
                    issue_types[name] = issue_types.get(name, 0) + 1

            if issue_types:
                st.markdown("#### 问题类型分布")
                issue_df = pd.DataFrame(list(issue_types.items()), columns=["维度", "出现次数"])
                fig = px.bar(issue_df, x="维度", y="出现次数",
                             color="维度",
                             color_discrete_sequence=color_palette[:len(issue_df)],
                             labels={"出现次数": "异常日出现次数"})
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=20),
                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 异常详情")

            sorted_anom = sorted(filtered, key=lambda x: x.get("drop", 0), reverse=True)

            for item in sorted_anom[:15]:
                date = item["date"]
                score = item.get("score", 0)
                drop = item.get("drop", 0)
                kw = item.get("keywords", [])
                dims = item.get("dimensions", [])

                if drop > 0.35:
                    css_class = "anomaly-critical"
                    level = "严重"
                    level_color = "#d94f4f"
                elif drop > 0.2:
                    css_class = "anomaly-warning"
                    level = "中等"
                    level_color = "#e8a030"
                else:
                    css_class = "anomaly-info"
                    level = "轻微"
                    level_color = "#4a8aaa"

                suggestions = generate_anomaly_suggestions(item)

                st.markdown(f"""
                <div class="{css_class}">
                    <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                        <span class="anomaly-date">{date}</span>
                        <span>
                            <span style="font-size:0.7rem; color:{level_color}; font-weight:600;">[{level}]</span>
                            <span class="anomaly-score">{score:.3f}</span>
                            <span class="anomaly-drop">降幅 {drop:.3f}</span>
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                if kw:
                    tags = " ".join([f'<span class="tag">{k["w"]}({k["c"]})</span>' for k in kw[:6]])
                    st.markdown(f'<div style="margin:0.2rem 0;">{tags}</div>', unsafe_allow_html=True)

                if dims:
                    txt = "、".join([f'{d["name"]}(当日{d["day"]:.2f}，全局{d["global"]:.2f})' for d in dims[:3]])
                    st.markdown(f'<div style="font-size:0.75rem; color:#5a6a6a;">问题维度：{txt}</div>',
                                unsafe_allow_html=True)

                if suggestions:
                    shtml = " ".join([f'<span class="suggestion">• {s}</span>' for s in suggestions[:3]])
                    st.markdown(f'<div style="margin-top:0.2rem;">{shtml}</div>', unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            if len(sorted_anom) > 15:
                st.caption(f"共 {len(sorted_anom)} 个异常日，仅显示前15个")


# ==================== 预测预警 ====================

elif page == "预测预警":
    st.markdown("### 未来30天预测")

    if forecast_df is not None:
        forecast_df["日期"] = pd.to_datetime(forecast_df["日期"])

        f_min = forecast_df["日期"].min().date()
        f_max = forecast_df["日期"].max().date()
        f_range = st.slider("预测日期范围", f_min, f_max, (f_min, f_max), format="YYYY-MM-DD")
        f_df = forecast_df[(forecast_df["日期"].dt.date >= f_range[0]) & (forecast_df["日期"].dt.date <= f_range[1])]

        if len(f_df) == 0:
            st.warning("当前日期范围内无数据")
        else:
            avgp = f_df["预测得分"].mean()
            minp = f_df["预测得分"].min()
            maxp = f_df["预测得分"].max()
            wcnt = f_df["预警"].sum()

            first = f_df["预测得分"].iloc[0]
            last = f_df["预测得分"].iloc[-1]
            if last - first > 0.04:
                trend = "上升"
                tcls = "badge-up"
            elif last - first < -0.04:
                trend = "下降"
                tcls = "badge-down"
            else:
                trend = "平稳"
                tcls = "badge-stable"

            st.markdown(f"""
            <div class="pred-stats">
                <div class="pred-item"><div class="pred-value" style="color:#2a7a5a;">{avgp:.3f}</div><div class="pred-label">预测均值</div></div>
                <div class="pred-item"><div class="pred-value" style="color:#4a8aaa;">{maxp:.3f}</div><div class="pred-label">最高点</div></div>
                <div class="pred-item"><div class="pred-value" style="color:#c94a4a;">{minp:.3f}</div><div class="pred-label">最低点</div></div>
                <div class="pred-item"><div class="pred-value" style="color:{'#c94a4a' if wcnt > 0 else '#2a8a6a'};">{wcnt}天</div><div class="pred-label">预警天数</div></div>
                <div class="pred-item"><div><span class="{tcls}">{trend}</span></div><div class="pred-label">趋势方向</div></div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=f_df["日期"], y=f_df["预测得分"],
                mode="lines+markers", name="预测值",
                line=dict(color="#d94f4f", width=2.5),
                marker=dict(size=5, color="#d94f4f")
            ))
            fig.add_trace(go.Scatter(
                x=f_df["日期"], y=f_df["上限"],
                mode="lines", name="置信区间",
                line=dict(color="#d94f4f", width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=f_df["日期"], y=f_df["下限"],
                mode="lines", name="置信区间",
                line=dict(color="#d94f4f", width=0),
                fill="tonexty", fillcolor="rgba(217,79,79,0.12)"
            ))
            fig.add_hline(y=0.45, line_dash="dash", line_color="#e8a030",
                          annotation_text="阈值 0.45", annotation_position="top right")

            wdf = f_df[f_df["预警"]]
            if len(wdf) > 0:
                fig.add_trace(go.Scatter(
                    x=wdf["日期"], y=wdf["预测得分"],
                    mode="markers", name="预警日",
                    marker=dict(size=10, color="#d94f4f", symbol="x")
                ))

            fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(showgrid=True, gridcolor='#eef3f1', tickformat='%m-%d'),
                              yaxis=dict(showgrid=True, gridcolor='#eef3f1', range=[0.3, 0.95]),
                              legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

            if wcnt > 0:
                days = f_df[f_df["预警"]]["日期"].dt.strftime('%Y-%m-%d').tolist()
                st.warning(f" {wcnt} 个预警日：{', '.join(days)}")
            else:
                st.success("未来30天无预警")

            with st.expander("查看预测数据"):
                st.dataframe(f_df, use_container_width=True, hide_index=True)
    else:
        st.info("未找到预测数据")


# ==================== 用户画像 ====================

elif page == "用户画像":
    st.markdown("### 用户画像分析")
    st.caption("基于游客评论行为的用户特征分析")

    profile = prepare_user_profile(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 游客来源地分布")
        if profile["locations"] is not None and len(profile["locations"]) > 0:
            loc_df = profile["locations"]
            fig = px.bar(
                loc_df,
                x="来源地",
                y="评论数",
                color="来源地",
                color_discrete_sequence=color_palette[:len(loc_df)],
                labels={"评论数": "评论数"},
                text_auto=True
            )
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                xaxis=dict(tickfont=dict(size=10))
            )
            fig.update_traces(textposition='outside', textfont=dict(size=11))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"共 {len(loc_df)} 个主要来源地，其余地区合并统计")
        else:
            st.info("暂无来源地数据")

    with col2:
        st.markdown("#### 游客类型分布")
        if profile["types"] is not None and len(profile["types"]) > 0:
            type_df = profile["types"]
            fig = px.pie(
                type_df,
                values="评论数",
                names="游客类型",
                color="游客类型",
                color_discrete_sequence=color_palette[:len(type_df)],
                hole=0.3
            )
            fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="v", y=0.5, x=1.05)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无游客类型数据")

    st.markdown("---")
    st.markdown("#### 用户画像摘要")

    summary_cols = st.columns(3)

    with summary_cols[0]:
        unique_users = df["userName"].nunique() if "userName" in df.columns else 0
        st.metric("独立评论用户", unique_users)

    with summary_cols[1]:
        if profile["locations"] is not None:
            st.metric("来源地覆盖", len(profile["locations"]))
        else:
            st.metric("来源地覆盖", "暂无数据")

    with summary_cols[2]:
        if profile["types"] is not None:
            st.metric("游客类型数", len(profile["types"]))
        else:
            st.metric("游客类型数", "暂无数据")


# ==================== 竞争对比（修改为二选对比） ====================

else:  # page == "竞争对比"
    st.markdown("### 竞品对比分析")
    st.caption("选择两个景区进行一对一对比，雷达图显示五维度得分差异")

    if compare_df is not None and len(compare_df) > 1:
        # 获取所有景区名称列表
        all_scenic_names = compare_df["景区"].tolist()
        
        # 两个选择框
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            scenic_a = st.selectbox("选择景区 A", all_scenic_names, index=0)
        with col_sel2:
            # 默认选择第二个（如果列表长度>1，选索引1，否则选0）
            default_b = 1 if len(all_scenic_names) > 1 else 0
            scenic_b = st.selectbox("选择景区 B", all_scenic_names, index=default_b)
        
        # 如果相同则提示
        if scenic_a == scenic_b:
            st.warning("请选择两个不同的景区进行对比")
        else:
            # 筛选两个景区的数据
            df_pair = compare_df[compare_df["景区"].isin([scenic_a, scenic_b])]
            if len(df_pair) < 2:
                st.error("未找到所选景区的完整数据")
            else:
                # 五维度雷达图
                dims = ["风景_得分", "服务_得分", "餐饮_得分", "住宿_得分", "交通_得分"]
                labels = ["风景", "服务", "餐饮", "住宿", "交通"]
                
                fig = go.Figure()
                colors = ['#2a7a5a', '#c94a4a']  # 两种不同颜色
                for i, (_, row) in enumerate(df_pair.iterrows()):
                    values = [row.get(d, 0) for d in dims]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels,
                        name=row.get("景区", "未知"),
                        fill='toself',
                        fillcolor=f'rgba({60 + i * 160}, {140 - i * 40}, {100 - i * 20}, 0.15)',
                        line=dict(width=2, color=colors[i % len(colors)])
                    ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
                        angularaxis=dict(tickfont=dict(size=12, weight=600))
                    ),
                    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                    height=380,
                    margin=dict(l=40, r=40, t=20, b=50),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 指标对比表格
                st.subheader("核心指标对比")
                cols_show = ["景区", "评论数", "情感得分均值", "评分均值", "正面占比", "负面占比"]
                avail_cols = [c for c in cols_show if c in df_pair.columns]
                st.dataframe(df_pair[avail_cols], use_container_width=True, hide_index=True)
                
                # 自动分析差距
                st.subheader("维度差距分析")
                row_a = df_pair[df_pair["景区"] == scenic_a].iloc[0]
                row_b = df_pair[df_pair["景区"] == scenic_b].iloc[0]
                
                diff_data = []
                for dim in dims:
                    val_a = row_a.get(dim, 0)
                    val_b = row_b.get(dim, 0)
                    diff = val_a - val_b
                    diff_data.append({
                        "维度": dim.replace("_得分", ""),
                        scenic_a: f"{val_a:.3f}",
                        scenic_b: f"{val_b:.3f}",
                        "差值 (A-B)": f"{diff:+.3f}",
                        "优势方": "A" if diff > 0 else ("B" if diff < 0 else "持平")
                    })
                diff_df = pd.DataFrame(diff_data)
                st.dataframe(diff_df, use_container_width=True, hide_index=True)
                
                # 找出差距最大的维度
                max_diff_dim = max(diff_data, key=lambda x: abs(float(x["差值 (A-B)"])))
                st.caption(f"📊 差距最大的维度：**{max_diff_dim['维度']}**（{scenic_a} 比 {scenic_b} {max_diff_dim['差值 (A-B)']}）")
                
                # 显示所有景区排名（折叠）
                with st.expander("查看所有景区排名"):
                    rank = compare_df.sort_values("情感得分均值", ascending=False).copy()
                    rank["排名"] = range(1, len(rank) + 1)
                    st.dataframe(rank[["排名", "景区", "情感得分均值", "正面占比"]], use_container_width=True, hide_index=True)
    else:
        st.info("未找到对比数据，请确保已运行 6_竞争对比与行动方案生成.py 生成汇总表")


# ==================== 页脚 ====================

st.markdown("---")
st.caption("乡村文旅数据智能分析平台  ·  数据来源：携程网公开评论  ·  仅供学习研究")
