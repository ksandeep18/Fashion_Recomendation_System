import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

from feature_extractor import extract_features

st.title('Fashion Recommender System')

# ensure uploads folder exists
os.makedirs('uploads', exist_ok=True)

# load precomputed data (fail fast with clear message)
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
    filenames = pickle.load(open('filenames.pkl','rb'))
except Exception as e:
    st.error("Missing or unreadable embeddings.pkl / filenames.pkl — run the preprocessing script (app.py) first.")
    st.stop()

def save_uploaded_file(uploaded_file):
    try:
        path = os.path.join('uploads', uploaded_file.name)
        with open(path,'wb') as f:
            f.write(uploaded_file.getbuffer())
        return path
    except Exception:
        return None

def recommend(features, feature_list, n_neighbors=6):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path:
        display_image = Image.open(saved_path)
        st.image(display_image, use_column_width=True)
        features = extract_features(saved_path)
        indices = recommend(features, feature_list)
        # show top-5 neighbors (skip the first if it's the same image)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            idx = indices[0][i+1] if indices.shape[1] > 1 else indices[0][i]
            col.image(filenames[idx], use_column_width=True)
    else:
        st.header("Error saving uploaded file")