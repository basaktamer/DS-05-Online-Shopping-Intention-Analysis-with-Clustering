import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# --- PAGE SETUP ---
st.set_page_config(page_title="Shopping Intention Analysis", layout="wide")
st.title("🛒 Online Shopping Intention Analysis")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv('online_shoppers_intention.csv')

try:
    df = load_data()
    features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']
    x = df[features]
    
    scaler = StandardScaler()
    # We use a DataFrame to keep Yellowbrick happy with feature names
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=features)

    # --- SIDEBAR ---
    st.sidebar.header("Model Settings")
    k_val = st.sidebar.slider("Select K for Silhouette", 2, 6, 2)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📉 Optimization", "🎯 Results", "📋 Data"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Elbow Method")
            # The Fix: Explicitly defining the model with n_init
            model_elbow = KMeans(n_init=10, random_state=42)
            # We explicitly set 'is_fitted=False' to help the visualizer
            visualizer = KElbowVisualizer(model_elbow, k=(2, 11), ax=plt.gca())
            visualizer.fit(x_scaled)
            visualizer.finalize()
            st.pyplot(plt.gcf())
            plt.close()
            
        with col2:
            st.subheader("2. Silhouette Visualizer")
            model_sil = KMeans(n_clusters=k_val, n_init=10, random_state=42)
            visualizer_sil = SilhouetteVisualizer(model_sil, colors='yellowbrick', ax=plt.gca())
            visualizer_sil.fit(x_scaled)
            visualizer_sil.finalize()
            st.pyplot(plt.gcf())
            plt.close()

    with tab2:
        st.subheader(f"Cluster Visualization (k={k_val})")
        final_model = KMeans(n_clusters=k_val, n_init=10, random_state=42)
        df['Cluster'] = final_model.fit_predict(x_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x='ProductRelated_Duration', y='Administrative_Duration', 
                        hue='Cluster', palette='viridis', ax=ax)
        st.pyplot(fig)

    with tab3:
        st.dataframe(df.head(100))

except Exception as e:
    st.error(f"Error: {e}")