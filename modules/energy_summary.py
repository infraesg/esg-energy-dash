
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import platform
import numpy as np
import plotly.express as px
from PIL import Image
from utils.filters import apply_filters
import plotly.graph_objects as go

def show():
    Image.MAX_IMAGE_PIXELS = None
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('dark_background')
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'

    st.header("📈 에너지 통계 및 지표")

    df = st.session_state["df"]
    if df is None or df.empty:
        st.error("⚠️ 데이터가 비어 있습니다.")
        return

    df_filtered = apply_filters(df)

    if "yymm" not in df_filtered.columns or "월사용량" not in df_filtered.columns or "실지급액" not in df_filtered.columns:
        st.error("필수 컬럼이 누락되어 있습니다.")
        return

    df_filtered["yymm"] = df_filtered["yymm"].astype(str).str.zfill(6)
    df_filtered["연도"] = df_filtered["yymm"].str[:4].astype(int)
    df_filtered["월"] = df_filtered["yymm"].str[4:6]

    df_filtered["월사용량"] = pd.to_numeric(df_filtered["월사용량"], errors="coerce")
    df_filtered["실지급액"] = df_filtered["실지급액"].astype(str).str.replace(",", "")
    df_filtered["실지급액"] = pd.to_numeric(df_filtered["실지급액"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["월사용량", "실지급액"])

    st.subheader("📊 요약 지표")
    summary = df_filtered.groupby("연도").agg(
        총사용량=("월사용량", "sum"),
        총요금=("실지급액", "sum")
    ).sort_index()
    summary["평균단가"] = summary["총요금"] / summary["총사용량"]

    # ✅ 연도 index를 2000년대 기준 보정
    summary.index = [2000 + int(i) if int(i) < 100 else int(i) for i in summary.index]

    if 2023 in summary.index and 2024 in summary.index:
        def gap(now, past):
            diff = now - past
            rate = diff / past * 100
            return f"{diff:,.0f} ({rate:+.1f}%)"

        col1, col2, col3 = st.columns(3)
        col1.metric("총 전기사용량 (kWh)", f"{summary.loc[2024, '총사용량']:,.0f}", gap(summary.loc[2024, '총사용량'], summary.loc[2023, '총사용량']))
        col2.metric("총 전기요금 (원)", f"{summary.loc[2024, '총요금']:,.0f}", gap(summary.loc[2024, '총요금'], summary.loc[2023, '총요금']))
        col3.metric("평균 단가 (원/kWh)", f"{summary.loc[2024, '평균단가']:,.1f}", gap(summary.loc[2024, '평균단가'], summary.loc[2023, '평균단가']))
    else:
        st.warning("2023 또는 2024년 데이터가 부족합니다.")
        st.write("summary index:", summary.index.tolist())

    st.subheader("📊 월별 사용량 & 요금 비교 (과년도대비)")
    monthly = df_filtered.groupby(["연도", "월"]).agg(
        월사용량=("월사용량", "sum"),
        실지급액=("실지급액", "sum")
    ).reset_index()
    monthly["단가"] = monthly["실지급액"] / monthly["월사용량"]

    # 모든 월 (1~12월) 기준으로 fill 0
    all_months = [f"{i:02d}" for i in range(1, 13)]
    usage_pivot = monthly.pivot(index="월", columns="연도", values="월사용량").reindex(all_months).fillna(0)
    cost_pivot = monthly.pivot(index="월", columns="연도", values="실지급액").reindex(all_months).fillna(0)

    usage_pivot.columns = [2000 + int(c) if int(c) < 100 else int(c) for c in usage_pivot.columns]
    cost_pivot.columns = [2000 + int(c) if int(c) < 100 else int(c) for c in cost_pivot.columns]

    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots()
        width = 0.35
        x = range(12)
        if 2023 in usage_pivot.columns and 2024 in usage_pivot.columns:
            ax1.bar([i - width/2 for i in x], usage_pivot[2023], width, label='2023', color='skyblue')
            ax1.bar([i + width/2 for i in x], usage_pivot[2024], width, label='2024', color='orange')
        elif 2023 in usage_pivot.columns:
            ax1.bar(x, usage_pivot[2023], width, label='2023', color='skyblue')
        elif 2024 in usage_pivot.columns:
            ax1.bar(x, usage_pivot[2024], width, label='2024', color='orange')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_months)
        ax1.set_title("월별 전기사용량 비교")
        ax1.set_xlabel("월")
        ax1.set_ylabel("사용량 (kWh)")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        width = 0.35
        x = range(12)
        if 2023 in cost_pivot.columns and 2024 in cost_pivot.columns:
            ax2.bar([i - width/2 for i in x], cost_pivot[2023], width, label='2023', color='skyblue')
            ax2.bar([i + width/2 for i in x], cost_pivot[2024], width, label='2024', color='orange')
        elif 2023 in cost_pivot.columns:
            ax2.bar(x, cost_pivot[2023], width, label='2023', color='skyblue')
        elif 2024 in cost_pivot.columns:
            ax2.bar(x, cost_pivot[2024], width, label='2024', color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_months)
        ax2.set_title("월별 전기요금 비교")
        ax2.set_xlabel("월")
        ax2.set_ylabel("요금 (원)")
        ax2.legend()
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        width = 0.35
        all_months = [f"{i:02d}" for i in range(1, 13)]
        for year in sorted(monthly["연도"].unique()):
            sub = monthly[monthly["연도"] == year].set_index("월").reindex(all_months)
            ax3.plot(all_months, sub["단가"], marker='o', label=str(year))
        ax3.set_xticks(range(12))
        ax3.set_xticklabels(all_months)
        ax3.set_title("평균 단가 추이 (원/kWh)")
        ax3.set_xlabel("월")
        ax3.set_ylabel("원/kWh")
        ax3.legend()
        st.pyplot(fig3)


    st.subheader("🍩 그룹별 전기사용량 비율")
    usage_col = "월사용량"
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏢 한전유형별**")
        hanjeon_map = {"D": "사옥", "L": "통합국", "A": "기지국", "M": "중계국"}
        hkeys = df_filtered["한전유형"].dropna().unique().tolist()
        hlabels = [hanjeon_map.get(k, "기타") for k in hkeys]
        hvalues = [df_filtered[df_filtered["한전유형"] == k][usage_col].sum() for k in hkeys]
        fig = go.Figure(data=[go.Pie(labels=hlabels, values=hvalues, hole=0.5, textinfo="label+percent")])
        fig.update_layout(annotations=[dict(text="한전유형", x=0.5, y=0.5, font_size=14, showarrow=False)], height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**⚡ 전기계약구분별**")
        labels = df_filtered["전기계약구분"].dropna().unique().tolist()
        values = [df_filtered[df_filtered["전기계약구분"] == k][usage_col].sum() for k in labels]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, textinfo="label+percent")])
        fig.update_layout(annotations=[dict(text="계약구분", x=0.5, y=0.5, font_size=14, showarrow=False)], height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**📃 계약종별 (상위 7)**")
        top7 = df_filtered["계약종별"].value_counts().nlargest(7).index.tolist()
        labels = top7
        values = [df_filtered[df_filtered["계약종별"] == k][usage_col].sum() for k in labels]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, textinfo="label+percent")])
        fig.update_layout(annotations=[dict(text="계약종별", x=0.5, y=0.5, font_size=14, showarrow=False)], height=350)
        st.plotly_chart(fig, use_container_width=True)

# 🔻 도넛차트 아래에 삽입 (기존 st.subheader("🍩 그룹별 전기사용량 비율") 이후)

    # -----------------------------
    # 🔌 트래픽당 전력소비 지표 (예시적)
    # -----------------------------
    st.subheader("🔌 트래픽당 전력소비 지표 (예시적)")

    # ✅ 예시 트래픽량 생성 (GB 단위)
    import numpy as np
    np.random.seed(42)
    df_filtered["총트래픽량_GB"] = np.random.uniform(500, 1500, size=len(df_filtered))  # 예: 월별 500~1500GB

    # ✅ 트래픽당 전력소비량 계산
    df_filtered["트래픽당소비전력"] = df_filtered["월사용량"] / df_filtered["총트래픽량_GB"]

    # ✅ 요약 카드 시각화
    col1, col2, col3 = st.columns(3)
    col1.metric("평균 소비전력 (kWh/GB)", f"{df_filtered['트래픽당소비전력'].mean():.2f}")
    col2.metric("전력당 처리량 (GB/kWh)", f"{(df_filtered['총트래픽량_GB'].sum() / df_filtered['월사용량'].sum()):.2f}")
    col3.metric("총 트래픽량 (GB)", f"{df_filtered['총트래픽량_GB'].sum():,.0f}")

    # ✅ 연도-월 파생
    df_filtered["yymm"] = df_filtered["yymm"].astype(str).str.zfill(6)
    df_filtered["연도"] = df_filtered["yymm"].str[:4]
    df_filtered["월"] = df_filtered["yymm"].str[4:]

    # ✅ 월별 평균 트래픽당 소비전력 추이 (꺾은선 그래프)
    import plotly.express as px
    grouped = df_filtered.groupby(["연도", "월"])["트래픽당소비전력"].mean().reset_index()

    fig_line = px.line(
        grouped, x="월", y="트래픽당소비전력", color="연도",
        title="📈 월별 트래픽당 전력소비량 (kWh/GB) – 예시값",
        markers=True, labels={"트래픽당소비전력": "kWh/GB"}
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ✅ 사업영역 본부명 매핑
    본부맵 = {
        "5100": "수도권본부",
        "5600": "중부본부",
        "5300": "동부본부",
        "5500": "서부본부"
    }
    df_filtered["사업영역_본부명"] = df_filtered["사업영역"].astype(str).map(본부맵)

    # ✅ 박스플롯 설명
    st.markdown("""
    📦 **사업영역별 트래픽당 전력소비량 박스플롯 (예시적)**  
    이 그래프는 각 본부 내 기지국들의 **트래픽당 전력소비량(kWh/GB)** 분포를 보여줍니다.  
     - **상자가 위로 길수록** 해당 지역 평균 소비전력이 높다는 뜻  
    """)

    # ✅ 박스플롯 시각화
    if "사업영역_본부명" in df_filtered.columns:
        fig_box = px.box(
            df_filtered, x="사업영역_본부명", y="트래픽당소비전력",
            labels={"트래픽당소비전력": "kWh/GB", "사업영역_본부명": "지역본부"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
