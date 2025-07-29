
import pandas as pd
import os
import streamlit as st

@st.cache_data
def load_usage_summary_data():
    """전기 사용량/요금 통계용 데이터 로드 (2023/2024 병합)"""
    base_path = "data"
    file_2023 = os.path.join(base_path, "monthly_power_data_2023.csv")
    file_2024 = os.path.join(base_path, "monthly_power_data_2024.csv")

    df_2023 = pd.read_csv(file_2023)
    df_2024 = pd.read_csv(file_2024)

    df_2023["연도"] = 2023
    df_2024["연도"] = 2024

    if "처리월" in df_2023.columns:
        df_2023.rename(columns={"처리월": "yymm"}, inplace=True)
    if "처리월" in df_2024.columns:
        df_2024.rename(columns={"처리월": "yymm"}, inplace=True)

    df_all = pd.concat([df_2023, df_2024], ignore_index=True)
    df_all["yymm"] = df_all["yymm"].astype(str).str.zfill(6)
    return df_all

@st.cache_data
def load_bill_vs_system_data():
    """청구서 vs 시스템 전력량 비교용 데이터 로드"""
    return pd.read_csv("data/zrew461_bill_vs_system.csv")


@st.cache_data
def load_detail_reference():
    """상세내용 데이터 로드"""
    return pd.read_csv("zpcode_model_all.csv")
