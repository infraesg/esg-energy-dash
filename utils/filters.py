import streamlit as st

def apply_filters(df):
    df_filtered = df.copy()

    # 📍 사업영역 필터
    biz_options = {
        "전체": None,
        "수도권": 5100,
        "동부": 5300,
        "서부": 5500,
        "중부": 5600
    }
    region = st.sidebar.selectbox("📍 지역본부", list(biz_options.keys()), index=0)
    if biz_options[region] is not None:
        df_filtered = df_filtered[df_filtered["사업영역"] == biz_options[region]]

    # 🏢 한전유형 필터
    hanjeon_map = {
        "전체": None,
        "사옥": "D",
        "통합국": "L",
        "기지국": "A",
        "중계국": "M",
        "기타": "기타"
    }
    hanjeon_type = st.sidebar.selectbox("🏢 국소유형", list(hanjeon_map.keys()), index=0)
    if hanjeon_type != "전체":
        if hanjeon_map[hanjeon_type] == "기타":
            df_filtered = df_filtered[~df_filtered["한전유형"].isin(["D", "L", "A", "M"])]
        else:
            df_filtered = df_filtered[df_filtered["한전유형"] == hanjeon_map[hanjeon_type]]

    # ⚡ 전기계약구분 필터
    contract_type = st.sidebar.selectbox("⚡ 전기계약구분", ["전체"] + sorted(df["전기계약구분"].dropna().unique()))
    if contract_type != "전체":
        df_filtered = df_filtered[df_filtered["전기계약구분"] == contract_type]

    # 🏷️ RAPA 여부 필터
    if "RAPA" in df_filtered.columns:
        rapa_option = st.sidebar.selectbox("🏷️ RAPA 여부", ["전체", "라파공용화국소", "일반국소"])
        if rapa_option == "라파공용화국소":
            df_filtered = df_filtered[df_filtered["RAPA"] == "X"]
        elif rapa_option == "일반국소":
            df_filtered = df_filtered[df_filtered["RAPA"] != "X"]

    return df_filtered
