import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Streamlit App ---

st.title('Customer Segmentation App')

# 1. Upload CSV
uploaded_file = st.file_uploader("Upload your cleaned dataset", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 2. Preprocessing
    st.subheader('Data Preprocessing')
    
    # Label Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Clustering
    st.subheader('Customer Segmentation (KMeans)')
    k = st.slider('Select number of clusters (k)', 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    df['Cluster'] = clusters
    
    # 4. Visualization
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Customer Segments Visualization')
    st.pyplot(fig)
    
    # 5. Show Cluster Data
    st.subheader('Cluster Summary')
    st.write(df['Cluster'].value_counts())

    # Option to download results
    st.download_button('Download Clustered Data', df.to_csv(index=False), file_name='clustered_customers.csv')

    st.header('Step 3: Download Clustered Data')

    # Provide download button
    st.download_button(
        label='Download Clustered Data as CSV',
        data=df.to_csv(index=False),
        file_name='customer_segments.csv',
        mime='text/csv'
    )
