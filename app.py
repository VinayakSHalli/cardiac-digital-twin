import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Cardiac Digital Twin", layout="wide")

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        # This reads your zip file
        df = pd.read_csv('results/hybrid_results.zip', compression='zip')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # 3. Sidebar
    st.sidebar.header("📊 Global Performance")
    st.sidebar.metric("Hybrid F1 Score", "55.4%")
    
    st.title("🫀 Cardiac Digital Twin")
    st.write("Phase-2 Evaluation")

    # 4. Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient Heart Rate", "74 BPM")
    with col2:
        avg_risk = round(df['Hybrid_Risk_Score'].mean() * 100, 1)
        st.metric("Avg Hybrid Risk", f"{avg_risk}%")
    with col3:
        st.metric("System Status", "Live")

    # 5. Chart
    st.subheader("📈 Risk Score Timeline")
    fig = px.line(df.head(500), y=['LSTM_Risk_Score', 'Hybrid_Risk_Score'], 
                  color_discrete_sequence=["#ff9999", "#ff4b4b"])
    st.plotly_chart(fig, use_container_width=True)

    # 6. THE TABLE (Simplified to avoid errors)
    st.subheader("🚨 Abnormal Heartbeats Detected")
    
    # Simple search for the column
    target_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['abnormal', 'label', 'target', 'class']):
            target_col = col
            break

    if target_col:
        # Show first 50 abnormal rows
        abnormal_df = df[df[target_col] == 1].head(50)
        
        # We use a very simple style that doesn't use complex CSS
        st.dataframe(abnormal_df.style.set_properties(**{
            'background-color': '#ffdbcc',
            'color': 'black',
            'font-weight': 'bold'
        }))
    else:
        st.write("Displaying raw data (No 'Abnormal' column found):")
        st.dataframe(df.head(20))
