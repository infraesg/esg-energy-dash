import streamlit as st
from utils.data_loader import load_usage_summary_data, load_bill_vs_system_data
from modules import energy_summary, anomaly_detection, bill_vs_system



st.set_page_config(page_title="에너지관리 대시보드", layout="wide")
st.title("⚡ Network Infra 에너지 관리 대시보드(Prototype)")

menu = st.sidebar.radio("📂 메뉴 선택", [
    "에너지 통계 및 지표",
    "청구서 기반 전기료 최적화(E-OS)",
    "청구서vs시스템 전력량비교"
])

# 메뉴에 따라 데이터 분기 로딩
try:
    if menu == "청구서vs시스템 전력량비교":
        df = load_bill_vs_system_data()
    else:
        df = load_usage_summary_data()
    st.session_state["df"] = df
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

# 페이지 호출
if menu == "에너지 통계 및 지표":
    energy_summary.show()
elif menu == "청구서 기반 전기료 최적화(E-OS)":
    anomaly_detection.show()
elif menu == "청구서vs시스템 전력량비교":
    bill_vs_system.show()
