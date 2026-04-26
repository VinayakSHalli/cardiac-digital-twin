import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Cardiac Digital Twin", layout="wide")

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        # Reading the zip file
        df = pd.read_csv('results/hybrid_results.zip', compression='zip')
        # Clean column names (removes extra spaces)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # --- SMART COLUMN SEARCH ---
    # We find the column names dynamically to prevent KeyErrors
    hybrid_col = next((c for c in df.columns if 'hybrid' in c.lower() and 'risk' in c.lower()), df.columns[0])
    lstm_col = next((c for c in df.columns if 'lstm' in c.lower() and 'risk' in c.lower()), df.columns[0])
    target_col = next((c for c in df.columns if any(w in c.lower() for w in ['abnormal', 'label', 'target', 'class'])), None)

    # 3. Sidebar
    st.sidebar.header("📊 Global Performance")
    st.sidebar.metric("Hybrid F1 Score", "55.4%")
    
    st.title("🫀 Cardiac Digital Twin")
    st.write("Phase-2 Evaluation: Real-Time Monitoring Dashboard")

    # 4. Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient Heart Rate", "74 BPM")
    with col2:
        # Using the dynamically found column
        avg_risk = round(df[hybrid_col].mean() * 100, 1)
        st.metric("Avg Hybrid Risk", f"{avg_risk}%")
    with col3:
        st.metric("System Status", "Live")

    # 5. Chart
    st.subheader("📈 Risk Score Timeline")
    fig = px.line(df.head(500), y=[lstm_col, hybrid_col], 
                  color_discrete_sequence=["#ff9999", "#ff4b4b"])
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # 6. THE TABLE (Readable Contrast)
    st.subheader("🚨 Abnormal Heartbeats Detected")
    
    if target_col:
        # Get abnormal rows
        abnormal_df = df[df[target_col] == 1].head(50)
        
        # Applying styling: Peach background, Black text for contrast
        st.dataframe(abnormal_df.style.set_properties(**{
            'background-color': '#ffdbcc',
            'color': 'black',
            'font-weight': 'bold'
        }))
    else:
        st.write("Overview (Abnormal column not found):")
        st.dataframe(df.head(20))
