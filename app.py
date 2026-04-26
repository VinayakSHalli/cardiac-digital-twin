import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Cardiac Digital Twin", layout="wide")

# 2. Data Loading (ONLY change: added zip support)
@st.cache_data
def load_data():
    try:
        # Reading the zip file you uploaded
        df = pd.read_csv('results/hybrid_results.zip', compression='zip')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # 3. Sidebar Metrics
    st.sidebar.header("📊 Global Performance")
    st.sidebar.metric("LSTM F1 Score", "37.7%")
    st.sidebar.metric("MLP F1 Score", "54.3%")
    st.sidebar.metric("Hybrid F1 Score", "55.4%")
    
    st.title("🫀 Cardiac Digital Twin: Real-Time Monitoring")
    st.write("Phase-2 Evaluation")

    # 4. Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient Heart Rate", "74 BPM")
    with col2:
        # Note: Ensure your CSV has this exact column name
        avg_risk = round(df['Hybrid_Risk_Score'].mean() * 100, 1)
        st.metric("Avg Hybrid Risk", f"{avg_risk}%")
    with col3:
        st.metric("System Latency", "14ms")

    # 5. Interactive Chart
    st.subheader("📈 Risk Score Timeline")
    fig = px.line(df.head(500), y=['LSTM_Risk_Score', 'Hybrid_Risk_Score'], 
                  color_discrete_sequence=["#ff9999", "#ff4b4b"])
    st.plotly_chart(fig, use_container_width=True)

    # 6. Data Table
    st.subheader("📋 Heartbeat Data Overview")
    st.dataframe(df.head(100))

else:
    st.warning("Please ensure hybrid_results.zip is in the /results folder.")
