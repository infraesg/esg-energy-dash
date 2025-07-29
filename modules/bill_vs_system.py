import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import io
from utils.filters import apply_filters

@st.cache_data
def load_data():
    file_path = os.path.join("data", "zrew461_bill_vs_system.csv")
    return pd.read_csv(file_path)

@st.cache_data
def load_detail_reference():
    file_path = os.path.join("data", "zpcode_model_all.csv")
    return pd.read_csv(file_path, dtype={"zpkcode": str, "yymm": str})

def show():
    st.header("📊 청구서 vs 시스템 전력량 비교")

    df = load_data()

    df["월사용량"] = pd.to_numeric(df["월사용량"], errors="coerce")
    df["시스템사용량"] = pd.to_numeric(df["시스템사용량"], errors="coerce")
    df = df.dropna(subset=["월사용량", "시스템사용량"])
    df = df[df["월사용량"] > 0]
    df["사용비율"] = df["시스템사용량"] / df["월사용량"] * 100
    df["yymm"] = df["yymm"].astype(str)

    df = apply_filters(df)

    col1, col2 = st.columns(2)
    with col1:
        sys_types = df["무슨시스템"].dropna().unique().tolist()
        system_filter = st.multiselect("무슨 시스템", sys_types, default=sys_types)
    with col2:
        upper_threshold = st.slider("과소청구 기준 (예: 97% 이상)", 0, 150, 97)
        lower_threshold = st.slider("과대청구 기준 (예: 83% 이하)", 0, 150, 83)

    df_filtered = df[df["무슨시스템"].isin(system_filter)]

    df_filtered["월별상태"] = df_filtered["사용비율"].apply(
        lambda x: "과소청구" if x >= upper_threshold else (
            "과대청구" if x <= lower_threshold else "정상"
        )
    )

    site_majority = (
        df_filtered.groupby("한전내역번호")["월별상태"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"월별상태": "청구상태"})
    )
    df_filtered = df_filtered.merge(site_majority, on="한전내역번호", how="left")

    total_bill = df_filtered["월사용량"].sum()
    total_sys = df_filtered["시스템사용량"].sum()
    avg_ratio = (df_filtered["사용비율"] * df_filtered["월사용량"]).sum() / df_filtered["월사용량"].sum()

    total_count = site_majority.shape[0]
    high_count = site_majority[site_majority["청구상태"] == "과소청구"].shape[0]
    low_count = site_majority[site_majority["청구상태"] == "과대청구"].shape[0]
    normal_count = site_majority[site_majority["청구상태"] == "정상"].shape[0]

    pct_high = high_count / total_count * 100 if total_count else 0
    pct_low = low_count / total_count * 100 if total_count else 0
    pct_norm = normal_count / total_count * 100 if total_count else 0

    st.subheader("✅ 요약 지표")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 청구서 전력량 (kWh)", f"{total_bill:,.0f}")
    col2.metric("총 시스템 전력량 (kWh)", f"{total_sys:,.0f}")
    col3.metric("청구서 전력량대비 시스템 전력량 (%)", f"{avg_ratio:.1f}%")
    col4.markdown(f"""
        **국소 분포:**  
        🔺 과소청구({upper_threshold}% 이상): {high_count} ({pct_high:.1f}%)  
        🔻 과대청구({lower_threshold}% 이하): {low_count} ({pct_low:.1f}%)  
        ✅ 정상 구간: {normal_count} ({pct_norm:.1f}%)
    """)

    st.subheader("📈 월별 및 시스템별 사용 현황")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**월별 전력량 비교**")
        month_sum = df_filtered.groupby("yymm")[["월사용량", "시스템사용량"]].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        width = 0.35
        x = range(len(month_sum))
        ax1.bar([i - width/2 for i in x], month_sum["월사용량"], width, label='청구서', color='orange')
        ax1.bar([i + width/2 for i in x], month_sum["시스템사용량"], width, label='시스템', color='skyblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(month_sum["yymm"].tolist(), rotation=45, fontsize=8)
        ax1.set_ylabel("전력량(kWh)", fontsize=8)
        ax1.legend(fontsize=6)
        st.pyplot(fig1, use_container_width=True)
    with col2:
        st.markdown("**월별 평균 사용비율 (%)**")
        diff_by_month = df_filtered.groupby("yymm")["사용비율"].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(diff_by_month["yymm"], diff_by_month["사용비율"], marker='o', color='crimson')
        ax2.axhline(y=upper_threshold, color='gray', linestyle='--', linewidth=1)
        ax2.axhline(y=lower_threshold, color='gray', linestyle='--', linewidth=1)
        ax2.set_ylabel("사용비율(%)", fontsize=8)
        ax2.set_xticklabels(diff_by_month["yymm"], rotation=45, fontsize=8)
        ax2.grid(True)
        st.pyplot(fig2, use_container_width=True)
    with col3:
        st.markdown("**시스템 유형별 평균 사용비율**")
        system_avg = df_filtered.groupby("무슨시스템")["사용비율"].mean().reset_index()
        fig3 = go.Figure(data=[
            go.Pie(labels=system_avg["무슨시스템"], values=system_avg["사용비율"], hole=0.5, textinfo="label+percent")
        ])
        fig3.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), annotations=[dict(text="사용비율", x=0.5, y=0.5, showarrow=False)])
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📋 과소청구 / 과대청구 국소 목록 (최신 yymm 기준)")

    df_high = df_filtered[df_filtered["청구상태"] == "과소청구"]
    latest_high = df_high.loc[df_high.groupby("한전내역번호")["yymm"].idxmax()]
    latest_high = latest_high.sort_values("사용비율", ascending=False)

    df_low = df_filtered[df_filtered["청구상태"] == "과대청구"]
    latest_low = df_low.loc[df_low.groupby("한전내역번호")["yymm"].idxmax()]
    latest_low = latest_low.sort_values("사용비율", ascending=True)

    col1, col2 = st.columns([6, 1])
    with col2:
        max_rows = st.selectbox("표시 개수", [5, 10, 20, 50, 100], index=1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### 🔺 과소청구 ({upper_threshold}% 이상)")
        header_cols = st.columns([2,1,1.5,1.5,1,1])
        header_cols[0].markdown("**한전번호**")
        header_cols[1].markdown("**사용연월**")
        header_cols[2].markdown("**청구서사용량**")
        header_cols[3].markdown("**시스템사용량**")
        header_cols[4].markdown("**사용비율**")
        header_cols[5].markdown("**상세보기**")  # ✅ 추가

        for i, row in latest_high.head(max_rows).iterrows():
            cols = st.columns([2, 1, 1.5, 1.5, 1, 1])
            cols[0].markdown(f"{row['한전내역번호']}")
            cols[1].write(row["yymm"])
            cols[2].write(f"{row['월사용량']:.0f}")
            cols[3].write(f"{row['시스템사용량']:.0f}")
            cols[4].write(f"{row['사용비율']:.1f}%")
            if cols[5].button("상세보기", key=f"btn_high_{i}"):
                st.session_state["selected_detail"] = {
                    "한전내역번호": row["한전내역번호"],
                    "yymm": row["yymm"],
                    "구분": "과소청구"
                }

    with col2:
        st.markdown(f"#### 🔻 과대청구 ({lower_threshold}% 이하)")
        header_cols = st.columns([2,1,1.5,1.5,1,1])
        header_cols[0].markdown("**한전번호**")
        header_cols[1].markdown("**사용연월**")
        header_cols[2].markdown("**청구서사용량**")
        header_cols[3].markdown("**시스템사용량**")
        header_cols[4].markdown("**사용비율**")
        header_cols[5].markdown("**상세보기**")  # ✅ 추가

        for i, row in latest_low.head(max_rows).iterrows():
            cols = st.columns([2, 1, 1.5, 1.5, 1, 1])
            cols[0].markdown(f"{row['한전내역번호']}")
            cols[1].write(row["yymm"])
            cols[2].write(f"{row['월사용량']:.0f}")
            cols[3].write(f"{row['시스템사용량']:.0f}")
            cols[4].write(f"{row['사용비율']:.1f}%")
            if cols[5].button("상세보기", key=f"btn_low_{i}"):
                st.session_state["selected_detail"] = {
                    "한전내역번호": row["한전내역번호"],
                    "yymm": row["yymm"],
                    "구분": "과대청구"
                }

    if "selected_detail" in st.session_state:
        st.markdown("### 🔍 상세정보")

        selected_id = str(st.session_state["selected_detail"]["한전내역번호"])
        selected_yymm = str(st.session_state["selected_detail"]["yymm"])

        # 원본 파일: zrew_bill_vs_system.csv 기반
        df["한전내역번호"] = df["한전내역번호"].astype(str)
        df["yymm"] = df["yymm"].astype(str)
        full_bill = df[df["한전내역번호"] == selected_id].copy()

        # 상세정보 파일: zpcode_model_all.csv 기반
        detail_df = load_detail_reference()
        detail_df["한전내역번호"] = detail_df["한전내역번호"].astype(str)
        detail_df["yymm"] = detail_df["yymm"].astype(str)
        detail_all = detail_df[detail_df["한전내역번호"] == selected_id].copy()

        if full_bill.empty:
            st.warning("📭 해당 국소의 청구서 기반 데이터가 없습니다.")
        else:
            st.markdown("### 📋 국소 사용 이력 상세")

            # 병합: yymm 기준으로 상세정보 붙이기
            merged_all = pd.merge(full_bill, detail_all, on=["한전내역번호", "yymm"], how="left")

            # 기준월 선택 옵션
            col1, col2 = st.columns([4, 1])
            with col2:
                yymm_options = merged_all["yymm"].unique().tolist()
                selected_yymm = st.selectbox("기준월 선택", options=sorted(yymm_options))
            with col1:
                st.markdown("")

            st.markdown("#### 🧾 사용량 요약")
            selected_rows = merged_all[merged_all["yymm"] == selected_yymm]
            if not selected_rows.empty:
                summary_row = selected_rows.iloc[0]
                summary_table = pd.DataFrame([{
                    "한전내역번호": selected_id,
                    "기준월(yymm)": selected_yymm,
                    "실사용시작일": summary_row.get("실사용시작일"),
                    "실사용종료일": summary_row.get("실사용종료일"),
                    "시스템 총합 (Wh)": summary_row.get("총합"),
                    "청구서 월사용량 (kWh)": summary_row.get("월사용량")
                }])
                st.dataframe(summary_table)
    
            st.markdown("#### 🧪 장비 및 추출 방식")
            detail_rows = merged_all[merged_all["yymm"] == selected_yymm].copy()

            def convert_logic(ems_val):
                if ems_val == "가능":
                    return "EMS"
                elif ems_val == "불가능":
                    return "PRB+시험성적서"
                else:
                    return "해당없음"

            detail_rows["추출로직"] = detail_rows["EMS"].apply(convert_logic)

            def extract_system(row):
                system_cols = ["시스템추출-3G", "시스템추출-4G_EMS", "시스템추출-4G_PRB(구)", "시스템추출-5G_EMS"]
                values = [str(row[col]) for col in system_cols if pd.notnull(row.get(col)) and row.get(col) != ""]
                return ", ".join(values)

            detail_rows["시스템 추출"] = detail_rows.apply(extract_system, axis=1)

            st.dataframe(detail_rows[["zpcode", "세대", "medel", "추출로직", "시스템 추출"]].rename(
                columns={"zpcode": "통시코드(zpcode)", "medel": "모델명"}
            ))

            Buffer = io.BytesIO()
            Buffer.write(merged_all.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))
            Buffer.seek(0)

            # ✅ 완전한 상세정보 CSV 다운로드
            st.download_button(
                "⬇️ 상세정보 CSV 다운로드",
                data=Buffer,
                file_name=f"상세정보_{selected_id}_전체기간.csv",
                mime="text/csv"
            )



            st.markdown("### 📊 월별 총합 전력량 & 청구서 사용량 비교")
            power_hist = merged_all[["yymm", "총합"]].copy()
            power_hist_unique = power_hist.groupby("yymm", as_index=False)["총합"].mean()

            bill_usage_by_month = full_bill.groupby("yymm", as_index=False)["월사용량"].sum()

            merged = pd.merge(power_hist_unique, bill_usage_by_month, on="yymm", how="left")

            fig, ax = plt.subplots()
            width = 0.35
            x = range(len(merged))
            ax.bar([i - width/2 for i in x], merged["총합"], width, label="시스템 총합(Wh)", color='green')
            ax.bar([i + width/2 for i in x], merged["월사용량"], width, label="청구서 사용량(kWh)", color='gray')
            ax.set_xticks(x)
            ax.set_xticklabels(merged["yymm"].tolist(), rotation=45)
            ax.set_ylabel("전력량")
            ax.legend()
            st.pyplot(fig)
