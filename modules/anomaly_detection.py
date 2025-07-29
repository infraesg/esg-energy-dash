import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils.filters import apply_filters

def show():
    st.header("⚠️ 이상국소 탐지 및 전기료 최적화")

    # ✅ 원본 데이터 불러오기
    df = st.session_state["df"]

    # ✅ 선택요금제 파생 (계약종별에서 /1, /2 추출)
    df["선택요금제"] = df["계약종별"].str.extract(r"/\s*(1|2)\s*$")[0].map({
        "1": "선택요금제 I",
        "2": "선택요금제 II"
    })

    # ✅ 필터 적용
    df_filtered = apply_filters(df)

    # ✅ 기본 수치 계산
    df_filtered["월사용량"] = pd.to_numeric(df_filtered["월사용량"], errors="coerce")
    df_filtered["실지급액"] = pd.to_numeric(df_filtered["실지급액"].astype(str).str.replace(",", ""), errors="coerce")
    df_filtered["단가"] = df_filtered["실지급액"] / df_filtered["월사용량"]
    df_filtered["yymm"] = df_filtered["yymm"].astype(str).str.zfill(4)

    # ✅ 계약전력 감설/증설용 평균 계산
    grouped = df_filtered.groupby("한전내역명칭")["월사용량"].mean().reset_index(name="평균사용량")
    merged = pd.merge(
        df_filtered.drop_duplicates("한전내역명칭")[["한전내역명칭", "계약전력"]],
        grouped,
        on="한전내역명칭"
    )
    merged["계약전력"] = pd.to_numeric(merged["계약전력"], errors="coerce")
    result = merged[merged["계약전력"] > 0].copy()

    # ✅ 탭 구성
    tab1, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 계약전력 감설/증설",
        "🔁 계약종별 변경 추천",
        "💰 정액제 이상",
        "❌ 사용량 0",
        "📊 선택요금제 변경"
    ])

    # 👉 이후 tab1~tab6 코드는 기존과 동일하게 유지하되,
    # 'tab6'은 선택요금제 컬럼이 df에 이미 존재하므로 추가 파생 없이 바로 사용하면 됩니다.

    with tab1:
        st.subheader("📊 계약전력 감설/증설 검토 후보")
        # ✅ 추가된 안내 메시지
        st.info("""
        ⚠️ **주의:**  
        이 분석은 **월평균 전력 사용량(kWh)** 기준으로 감설/증설 대상을 1차 탐지합니다.  
        하지만 통신장비는 **시간대별 트래픽 부하에 따라 전력소모가 크게 달라지므로**,  
        **최대 부하(피크 시간대)**나 **EMS 기반 실시간 소비 전력**을 반영한 정밀 분석이 필요합니다.  

        → 실제 감설 실행 시에는 **PS 기능 상태**, **시간별 peak 사용률**, **계약전력 초과 여부** 등을 반드시 함께 고려해야 합니다.
        """)
        
        # 🔧 기준 슬라이더
        down = st.slider("감설 기준 사용률 미만 (%)", 0, 100, 10, step=5)
        up = st.slider("증설 기준 사용률 초과 (%)", 100, 200, 110, step=5)

        # 🧠 설명
        st.markdown(f"""
        **💡 계약전력과 사용률의 개념**  
        - **계약전력 (kW)**: 한전과 약정한 최대 사용 전력  
        - **월사용량 (kWh)**: 해당 월 전체 사용 전력량 누계  
        - **평균부하 (kW)** = 월사용량 ÷ 720시간  
        - **사용률 (%)** = 평균부하 ÷ 계약전력 × 100  

        **✔️ 판정 기준:**  
        - 감설 검토 대상: 계약전력 **3kW 초과**이며, 사용률 **{down}% 미만**  
        - 증설 검토 대상: 사용률 **{up}% 초과**  
        - 그 외: 적정
        """)

        # 계산
        result["평균부하(kW)"] = result["평균사용량"] / 720
        result["사용률(%)"] = result["평균부하(kW)"] / result["계약전력"] * 100

        def classify(row):
            if row["계약전력"] > 3 and row["사용률(%)"] < down:
                return "감설 대상"
            elif row["사용률(%)"] > up:
                return "증설 대상"
            else:
                return "적정"
        result["판정결과"] = result.apply(classify, axis=1)

        # 요약 지표
        counts = result["판정결과"].value_counts()
        total_sites = len(result)
        low = counts.get("감설 대상", 0)
        high = counts.get("증설 대상", 0)
        normal = counts.get("적정", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 국소 수", f"{total_sites:,}개")
        col2.metric("감설 대상", f"{low:,}개")
        col3.metric("증설 대상", f"{high:,}개")
        col4.metric("적정", f"{normal:,}개")

        # 히스토그램
        st.markdown("### 📈 사용률(%) 분포 히스토그램")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(result["사용률(%)"], bins=30, color="skyblue", edgecolor="black")
        ax.axvline(x=down, color='red', linestyle='--', label=f"감설 기준 {down}%")
        ax.axvline(x=up, color='orange', linestyle='--', label=f"증설 기준 {up}%")
        ax.set_xlabel("사용률 (%)")
        ax.set_ylabel("국소 수")
        ax.set_title("사용률 분포 및 기준선")
        ax.legend()
        st.pyplot(fig)

        # 필터 선택
        filter_choice = st.multiselect("표시할 판정 결과", ["감설 대상", "증설 대상", "적정"], default=["감설 대상", "증설 대상"])
        filtered_result = result[result["판정결과"].isin(filter_choice)]

        # 테이블 출력
        cols = ["한전내역명칭", "계약전력", "평균사용량", "평균부하(kW)", "사용률(%)", "판정결과"]
        st.dataframe(filtered_result[cols].sort_values("사용률(%)"))


    with tab3:
        st.subheader("🔁 계약종별 변경 추천")

        # 📘 설명 문구
        st.info("""
        📌 **계약종별 변경 추천 기준**  
        최근 3개월간의 평균 월사용량(kWh)을 기준으로,  
        계약종별(주택용 ↔ 일반용)이 적절한지 판단하여 변경을 추천합니다.

        - `주택용`인데 평균 사용량이 **400kWh 초과** → 일반용 추천  
        - `일반용`인데 평균 사용량이 **380kWh 미만** → 주택용 추천

        ※ 최근 3개월 중 **2개월 이상 사용 데이터가 존재하는 국소**만 추천 대상으로 포함됩니다.
        """)

        # 🎚️ 슬라이더: 추천 판단 기준
        housing_cut = st.number_input("주택용 상한선 (kWh)", value=400)
        general_cut = st.number_input("일반용 하한선 (kWh)", value=380)

        # 🗓️ 최근 3개월 데이터 필터
        df_filtered["yymm"] = df_filtered["yymm"].astype(int)
        recent_months = sorted(df_filtered["yymm"].unique())[-3:]
        df_recent = df_filtered[df_filtered["yymm"].isin(recent_months)].copy()

        # ✅ 최근 3개월 중 2개월 이상 데이터가 있는 국소만 유지
        usage_counts = df_recent.groupby("한전내역명칭")["월사용량"].count()
        valid_sites = usage_counts[usage_counts >= 2].index.tolist()
        df_recent = df_recent[df_recent["한전내역명칭"].isin(valid_sites)]

        # 평균 사용량 계산
        avg_use = df_recent.groupby("한전내역명칭")["월사용량"].mean().reset_index(name="평균사용량")

        # 계약종별 병합
        joined = pd.merge(
            df_filtered[["한전내역명칭", "계약종별"]].drop_duplicates(),
            avg_use,
            on="한전내역명칭"
        )

        # 추천 판단 함수
        def get_recommendation(row):
            if "주택" in row["계약종별"] and row["평균사용량"] > housing_cut:
                return "일반용 추천", f"{row['평균사용량']:.1f}kWh > {housing_cut}"
            elif "일반" in row["계약종별"] and row["평균사용량"] < general_cut:
                return "주택용 추천", f"{row['평균사용량']:.1f}kWh < {general_cut}"
            else:
                return None, None

        # 추천 방향/사유 생성
        joined["추천방향"], joined["추천사유"] = zip(*joined.apply(get_recommendation, axis=1))
        recommend = joined[joined["추천방향"].notnull()]
        total_sites = len(joined)

        # ✅ 추천 방향별 개수
        housing_to_general = recommend[recommend["추천방향"] == "일반용 추천"].shape[0]
        general_to_housing = recommend[recommend["추천방향"] == "주택용 추천"].shape[0]
        nochange_sites = total_sites - (housing_to_general + general_to_housing)

        # ✅ 요약 테이블
        st.markdown("### 📋 계약종별 추천 요약")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("총 국소 수", f"{total_sites:,}개")
        col2.metric("주택용 → 일반용 추천", f"{housing_to_general:,}개")
        col3.metric("일반용 → 주택용 추천", f"{general_to_housing:,}개")
        col4.metric("종별 유지", f"{nochange_sites:,}개")


        # ✅ 시각화: 파이 + 막대 (1행 2열)
        col1, col2 = st.columns(2)

        with col1:
            pie_df = pd.DataFrame({
                "구분": ["주택→일반", "일반→주택", "유지"],
                "값": [housing_to_general, general_to_housing, nochange_sites]
            })
            fig1 = px.pie(pie_df, names="구분", values="값", title="계약종별 추천 분포", hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            bar_df = recommend["추천방향"].value_counts().reset_index()
            bar_df.columns = ["추천방향", "국소 수"]
            fig2 = px.bar(bar_df, x="추천방향", y="국소 수", text="국소 수", title="추천 방향별 국소 수")
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # ✅ 추천 대상 필터
        st.markdown("#### 🔍 추천 대상 목록")
        selected_directions = st.multiselect(
            "표시할 추천 방향",
            options=["주택용 추천", "일반용 추천"],
            default=["주택용 추천", "일반용 추천"]
        )

        filtered_recommend = recommend[recommend["추천방향"].isin(selected_directions)]

        # ✅ 추천 리스트 테이블
        st.dataframe(
            filtered_recommend[["한전내역명칭", "계약종별", "평균사용량", "추천방향", "추천사유"]]
            .sort_values("평균사용량")
            .reset_index(drop=True)
        )



    with tab4:
        st.subheader("💰 정액제 이상 의심 국소 탐지")

        st.info("""
        📌 **탐지 목적**  
        본 기능은 청구서 기반 정보만으로 **정액제 이상 가능성이 있는 국소를 탐지**합니다.  
        실제 사용량을 알 수 없으므로, **단가(원/kWh)가 비정상적으로 높거나 낮은 경우를 의심 국소로 분류**합니다.

        ※ 의심 국소는 현장 확인 또는 계약정보 점검이 필요합니다.
        """)

        # 필터링: 정액제만
        df_fixed = df_filtered[df_filtered["전기계약구분"].str.contains("정액제", na=False)].copy()
        df_fixed["단가"] = df_fixed["실지급액"] / df_fixed["월사용량"]
        df_fixed.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_fixed.dropna(subset=["단가"], inplace=True)

        # 임계값 슬라이더
        threshold = st.slider("이상 단가 기준 (원/kWh)", min_value=100, max_value=1000, step=10, value=300)

        # 이상 여부 판별
        df_fixed["의심여부"] = df_fixed["단가"] > threshold

        # 요약 지표
        total_fixed = len(df_fixed)
        suspicious = df_fixed["의심여부"].sum()
        normal = total_fixed - suspicious

        col1, col2, col3 = st.columns(3)
        col1.metric("총 정액제 국소", f"{total_fixed:,}개")
        col2.metric("의심 국소 수", f"{suspicious:,}개")
        col3.metric("정상 국소 수", f"{normal:,}개")

        # 시각화
        chart_data = pd.DataFrame({
            "구분": ["의심", "정상"],
            "개수": [suspicious, normal]
        })

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(chart_data, names="구분", values="개수", hole=0.4, title="단가 기준 이상 비율"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(chart_data, x="구분", y="개수", title="이상 여부 분포"), use_container_width=True)

        # 필터
        st.markdown("### 🔍 의심 국소 목록")
        selected_types = st.multiselect(
            "전기계약구분 필터",
            options=sorted(df_fixed["전기계약구분"].unique()),
            default=sorted(df_fixed["전기계약구분"].unique())
        )
        df_result = df_fixed[df_fixed["전기계약구분"].isin(selected_types)]

        # 데이터 테이블 출력
        st.dataframe(
            df_result[["한전내역번호", "yymm", "전기계약구분", "월사용량", "실지급액", "단가", "의심여부"]]
            .sort_values("단가", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

    with tab5:
        st.markdown("## 🔌 사용량 0 국소")

        st.info("""
        📌 본 탭은 **가장 최근 청구월 기준으로 '월사용량이 0'인 국소** 중에서  
        `실지급액 > 0` 인 경우를 **의심 국소**로 탐지합니다.

        - 계약상 정액제일 수 있으나, **정확한 사용량 정보가 없다면 현장 확인 필요**  
        - '정상'은 사용량 0이고 실지급액도 0인 경우
        """)

        # ✅ 가장 최근 월 기준 필터
        latest_yymm = df["yymm"].max()
        df_latest = df[df["yymm"] == latest_yymm].copy()

        # ✅ 데이터 타입 변환 (문자열일 경우 대비)
        df_latest["월사용량"] = pd.to_numeric(df_latest["월사용량"], errors="coerce")
        df_latest["실지급액"] = pd.to_numeric(df_latest["실지급액"], errors="coerce")

        # ✅ 상태 판별
        df_latest["사용량 0 여부"] = df_latest["월사용량"] == 0
        df_latest["상태"] = np.where(
            (df_latest["월사용량"] == 0) & (df_latest["실지급액"] > 0),
            "의심",
            "정상"
        )

        # ✅ 요약 카드 출력
        total_count = len(df_latest)
        zero_count = len(df_latest[df_latest["월사용량"] == 0])
        suspicious_count = len(df_latest[df_latest["상태"] == "의심"])

        col1, col2, col3 = st.columns(3)
        col1.metric("총 국소 수", f"{total_count:,}개")
        col2.metric("사용량 0 국소", f"{zero_count:,}개")
        col3.metric("의심 대상", f"{suspicious_count:,}개")

        # ✅ 도넛 그래프
        pie_df = pd.DataFrame({
            "구분": ["의심", "정상"],
            "개수": [
                (df_latest["상태"] == "의심").sum(),
                (df_latest["상태"] == "정상").sum()
            ]
        })
        fig = px.pie(pie_df, names="구분", values="개수", title="사용량 0 국소 분포", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        # ✅ 필터 위젯
        상태선택 = st.multiselect(
            "표시할 상태 선택",
            options=["의심", "정상"],
            default=["의심"]
        )

        # ✅ 조건 필터링
        df_filtered = df_latest[
            (df_latest["사용량 0 여부"]) & (df_latest["상태"].isin(상태선택))
        ].copy()

        # ✅ 출력 컬럼
        출력컬럼 = ["한전내역명칭", "yymm", "계약전력", "월사용량", "실지급액", "상태"]
        st.dataframe(df_filtered[출력컬럼], use_container_width=True)

        # ✅ 다운로드 버튼
        csv = df_filtered[출력컬럼].to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 CSV 다운로드",
            data=csv,
            file_name=f"사용량0_의심국소_{latest_yymm}.csv",
            mime="text/csv"
        )

    with tab6:
        st.subheader("📊 선택 요금제 변경 추천")

        # ✅ 항상 설명 표시
        st.info("""
        📌 **선택요금제 변경 추천 기준**  
        최근 3개월간 평균 사용량 기준으로, 현재 요금제가 적절한지 판단합니다.

        - `선택요금제 I`인데 평균 사용량이 **기준 초과** → II 추천  
        - `선택요금제 II`인데 평균 사용량이 **기준 미만** → I 추천  

        ※ 최근 3개월 중 **2개월 이상 데이터가 있는 국소만 분석**합니다.
        """)

        # ▶ 선택요금제 현황 요약 출력
        전체선택요금제 = df["선택요금제"].notna().sum()
        필터후선택요금제 = df_filtered["선택요금제"].notna().sum()

        # 최근 3개월 기준 유효 국소 수 계산
        df_filtered["yymm"] = df_filtered["yymm"].astype(int)
        recent_3 = sorted(df_filtered["yymm"].unique())[-3:]
        df_recent = df_filtered[df_filtered["yymm"].isin(recent_3)].copy()
        usage_counts = df_recent.groupby("한전내역명칭")["월사용량"].count()
        유효국소수 = (usage_counts >= 2).sum()

        # 💬 요약 카드
        st.info(f"""
        - 전체 데이터 내 선택요금제 국소 수: **{전체선택요금제:,}개**  
        - 현재 필터 조건 적용 후 선택요금제 국소 수: **{필터후선택요금제:,}개**  
        - 최근 3개월 중 2개월 이상 데이터 존재하는 국소 수: **{유효국소수:,}개**
        """)

        # ✅ 조건 미충족 시 메시지
        if 유효국소수 == 0:
            st.warning("현재 선택한 조건에서는 분석 가능한 선택요금제 국소가 없습니다.")
        else:

            # ▶️ 기준값 입력
            기준값 = st.number_input("선택요금제 변경 판단 기준 (kWh)", min_value=100, max_value=1000, value=460, step=10)

            # ▶️ 최근 3개월 필터
            df_filtered["yymm"] = df_filtered["yymm"].astype(int)
            recent_3 = sorted(df_filtered["yymm"].unique())[-3:]
            df_recent = df_filtered[df_filtered["yymm"].isin(recent_3)].copy()

            # ▶️ 유효 국소 추출: 최근 3개월 중 2개월 이상 존재
            count_check = df_recent.groupby("한전내역명칭")["월사용량"].count()
            valid_sites = count_check[count_check >= 2].index
            df_recent = df_recent[df_recent["한전내역명칭"].isin(valid_sites)].copy()

            # ▶️ 평균 사용량 + 선택요금제 병합
            avg_df = df_recent.groupby("한전내역명칭")["월사용량"].mean().reset_index(name="평균사용량")
            plan_df = df_recent[["한전내역명칭", "선택요금제"]].drop_duplicates()
            merged = pd.merge(plan_df, avg_df, on="한전내역명칭")

            # ▶️ 추천 판단 로직
            def 판단(row):
                if row["선택요금제"] == "선택요금제 I" and row["평균사용량"] > 기준값:
                    return "선택요금제 II 추천", f"{row['평균사용량']:.1f}kWh > {기준값}"
                elif row["선택요금제"] == "선택요금제 II" and row["평균사용량"] < 기준값:
                    return "선택요금제 I 추천", f"{row['평균사용량']:.1f}kWh < {기준값}"
                else:
                    return None, None

            # ▶️ 추천 결과 생성
            merged["추천방향"], merged["추천사유"] = zip(*merged.apply(판단, axis=1))
            추천대상 = merged[merged["추천방향"].notnull()]
            총국소 = len(merged)
            추천수 = len(추천대상)
            유지수 = 총국소 - 추천수

            # ▶️ 요약 카드
            col1, col2, col3 = st.columns(3)
            col1.metric("총 국소 수", f"{총국소:,}개")
            col2.metric("변경 추천", f"{추천수:,}개")
            col3.metric("유지", f"{유지수:,}개")

            # ▶️ 시각화
            col1, col2 = st.columns(2)
            with col1:
                pie = px.pie(
                    names=["추천", "유지"],
                    values=[추천수, 유지수],
                    title="요금제 추천 분포",
                    hole=0.4
                )
                st.plotly_chart(pie, use_container_width=True)

            with col2:
                bar = 추천대상["추천방향"].value_counts().reset_index()
                bar.columns = ["추천방향", "국소 수"]
                fig = px.bar(bar, x="추천방향", y="국소 수", text="국소 수", title="추천 방향별 분포")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            # ▶️ 추천 목록 필터
            st.markdown("### 🔍 추천 대상 목록")
            선택 = st.multiselect("표시할 추천 방향", 추천대상["추천방향"].unique().tolist(), default=추천대상["추천방향"].unique().tolist())
            필터링 = 추천대상[추천대상["추천방향"].isin(선택)]

            # ▶️ 결과 테이블
            st.dataframe(
                필터링[["한전내역명칭", "선택요금제", "평균사용량", "추천방향", "추천사유"]]
                .sort_values("평균사용량")
                .reset_index(drop=True),
                use_container_width=True
            )
