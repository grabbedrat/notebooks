import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Bookmark Clustering")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
import plotly.express as px
import base64

# Load data
@st.cache_data
def load_data():
    with open('embedded_bookmarks.json') as file:
        bookmarks = json.load(file)
    embeddings = np.array([bookmark["embedding"] for bookmark in bookmarks])
    return bookmarks, embeddings

bookmarks, embeddings = load_data()

# Streamlit app
st.title('Enhanced HDBSCAN Clustering for Bookmarks')

# Sidebar for parameters
st.sidebar.header('Clustering Parameters')
min_cluster_size = st.sidebar.slider('Min Cluster Size', 2, 20, 5)
min_samples = st.sidebar.slider('Min Samples', 1, 10, 1)
cluster_selection_epsilon = st.sidebar.slider('Cluster Selection Epsilon', 0.0, 1.0, 0.0)
metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'cosine', 'manhattan'])

# Dimensionality reduction
dim_reduction = st.sidebar.selectbox('Dimensionality Reduction', ['PCA', 'UMAP'])
n_components = st.sidebar.slider('Number of Components', 2, 10, 2)

# Preprocessing
normalize = st.sidebar.checkbox('Normalize Data', value=True)

@st.cache_data
def preprocess_and_reduce(embeddings, dim_reduction, n_components, normalize):
    if normalize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    if dim_reduction == 'PCA':
        reducer = PCA(n_components=n_components)
    elif dim_reduction == 'UMAP':
        import umap
        reducer = umap.UMAP(n_components=n_components)
    
    reduced_features = reducer.fit_transform(embeddings)
    return reduced_features

reduced_features = preprocess_and_reduce(embeddings, dim_reduction, n_components, normalize)

# Clustering
@st.cache_data
def perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric
    )
    clusterer.fit(reduced_features)
    return clusterer

clusterer = perform_clustering(reduced_features, min_cluster_size, min_samples, cluster_selection_epsilon, metric)

# Visualization
st.header('Clustering Visualization')
df = pd.DataFrame(reduced_features[:, :2], columns=['Component 1', 'Component 2'])
df['Cluster'] = clusterer.labels_
df['Title'] = [b['title'] for b in bookmarks]

fig = px.scatter(df, x='Component 1', y='Component 2', color='Cluster', hover_data=['Title'],
                 title='HDBSCAN Clustering Results')
st.plotly_chart(fig, use_container_width=True)

# Clustering statistics
n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
n_noise = list(clusterer.labels_).count(-1)

col1, col2 = st.columns(2)
with col1:
    st.metric("Number of Clusters", n_clusters)
with col2:
    st.metric("Number of Noise Points", n_noise)

# Display cluster contents
st.header('Cluster Contents')
cluster_df = pd.DataFrame({
    'Title': [b['title'] for b in bookmarks],
    'URL': [b['url'] for b in bookmarks],
    'Cluster': clusterer.labels_
})

for cluster in sorted(set(clusterer.labels_)):
    if cluster == -1:
        st.subheader('Noise Points')
    else:
        st.subheader(f'Cluster {cluster}')
    
    cluster_items = cluster_df[cluster_df['Cluster'] == cluster][['Title', 'URL']]
    st.dataframe(cluster_items)

# Generate prompts
st.header('Generated Prompts')

def generate_prompts(cluster_id, indent=""):
    prompts = []
    folder_name = f"Folder {cluster_id}"
    bookmarks_in_cluster = [b for b, label in zip(bookmarks, clusterer.labels_) if label == cluster_id]
    
    if bookmarks_in_cluster:
        bookmark_titles = [bookmark["title"] for bookmark in bookmarks_in_cluster]
        bookmark_titles_str = "\n".join([f"{indent}  - {title}" for title in bookmark_titles])
        prompt = f"{indent}Folder: {folder_name}\n{bookmark_titles_str}"
        prompts.append(prompt)
    
    return prompts

all_prompts = []
for cluster in sorted(set(clusterer.labels_)):
    if cluster != -1:  # Exclude noise points
        cluster_prompts = generate_prompts(cluster)
        all_prompts.extend(cluster_prompts)
        for prompt in cluster_prompts:
            st.text(prompt)
            st.text("")  # Add an empty line for readability

# Download prompts
if all_prompts:
    prompt_text = "\n\n".join(all_prompts)
    st.download_button(
        label="Download Prompts",
        data=prompt_text,
        file_name="prompts.txt",
        mime="text/plain"
    )