# app.py
import streamlit as st
import pandas as pd
import joblib

# 1) 모델 로드 ------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_watch_model.pkl")  # 같은 폴더에 있어야 함
    return model

model = load_model()

# 2) 페이지 설정 ----------------------------------------------
st.set_page_config(
    page_title="스마트워치 건강 위험 예측",
    page_icon="⌚",
    layout="centered"
)

st.title("⌚ 스마트워치 기반 건강 위험 예측 모델")
st.markdown("스마트워치의 생체 데이터를 기반으로 저산소증 위험 여부를 예측합니다.")

st.divider()

# 3) 입력 UI --------------------------------------------------
st.subheader("📥 생체 데이터 입력")

col1, col2 = st.columns(2)

with col1:
    heart_rate = st.number_input("💓 심박수 (BPM)", min_value=30, max_value=200, value=80)
    spo2 = st.number_input("🫁 산소포화도 SpO₂ (%)", min_value=80.0, max_value=100.0, value=97.0)
    steps = st.number_input("🚶 걸음 수", min_value=0, max_value=50000, value=5000)

with col2:
    sleep = st.number_input("😴 수면 시간 (시간)", min_value=0.0, max_value=15.0, value=7.0)
    stress = st.number_input("😰 스트레스 지수 (1~10)", min_value=1, max_value=10, value=3)
    activity = st.selectbox("🏃 활동 수준", ["Sedentary", "Active", "Highly Active"])

st.divider()

# 4) 입력 데이터 DataFrame 구성 -------------------------------
input_df = pd.DataFrame({
    "Heart Rate (BPM)": [heart_rate],
    "Blood Oxygen Level (%)": [spo2],
    "Step Count": [steps],
    "Sleep Duration (hours)": [sleep],
    "Stress Level": [stress],
    "Activity Level": [activity]
})

# 5) 예측 & 위험 판단 ----------------------------------------
st.subheader("📊 예측 실행")

if st.button("🩺 건강 상태 예측하기"):

    # 모델 예측
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)

    # 의학적 기준
    danger_spo2 = spo2 < 95         # SpO2 95% 미만
    danger_hr = heart_rate > 100    # HR 100BPM 초과

    # 최종 위험 여부
    final_alert = (pred == 1) or danger_spo2 or danger_hr

    st.write(f"### 🔢 예측된 위험 확률: **{proba:.3f}**")

    if final_alert:
        st.error("🚨 위험 신호 감지! 주의가 필요합니다.")
        st.markdown("#### ⚠️ 위험 요인")
        if pred == 1:
            st.markdown("- AI 모델이 **위험 상태**로 예측했습니다.")
        if danger_spo2:
            st.markdown("- 산소포화도가 **95% 미만**입니다.")
        if danger_hr:
            st.markdown("- 심박수가 **100BPM 이상**입니다.")
    else:
        st.success("✅ 현재 생체 데이터는 정상 범위입니다.")

    with st.expander("📋 입력 데이터 확인"):
        st.write(input_df)

st.markdown("---")
st.caption("© 2025 스마트헬스 AI 팀 | Random Forest 기반 저산소증 예측 시스템")
