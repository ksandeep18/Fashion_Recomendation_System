import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import os
from tqdm import tqdm

# ---------------------------
# 1. Build ResNet50 + Pooling Model
# ---------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ---------------------------
# 2. Helper: Safe Normalization
# ---------------------------
def normalize(vectors, eps=1e-10):
    """L2-normalize feature vectors row-wise"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + eps)

# ---------------------------
# 3. Batch Feature Extraction
# ---------------------------
def extract_batch_features(filepaths, model, batch_size=32):
    features = []
    valid_files = []

    for i in tqdm(range(0, len(filepaths), batch_size), desc="Extracting features"):
        batch_paths = filepaths[i:i+batch_size]
        imgs = []
        for path in batch_paths:
            try:
                img = image.load_img(path, target_size=(224,224))
                arr = image.img_to_array(img)
                imgs.append(arr)
                valid_files.append(path)
            except Exception as e:
                print(f" Skipped {path} due to error: {e}")
        
        if imgs:
            batch = np.stack(imgs, axis=0)
            batch = preprocess_input(batch)
            preds = model.predict(batch, verbose=0)
            features.append(preds)

    features = np.vstack(features)
    features = normalize(features)   # L2 normalization
    return features, valid_files

# ---------------------------
# 4. Collect image files
# ---------------------------
image_dir = "images"
filepaths = [
    os.path.join(image_dir, f) 
    for f in os.listdir(image_dir) 
    if f.lower().endswith((".jpg",".jpeg",".png"))
]

# ---------------------------
# 5. Run extraction
# ---------------------------
feature_array, valid_files = extract_batch_features(filepaths, model, batch_size=32)

# ---------------------------
# 6. Save embeddings + filenames
# ---------------------------
np.save("embeddings.npy", feature_array)   # shape (N, 2048)
np.save("filenames.npy", np.array(valid_files))

print(" Saved embeddings.npy and filenames.npy")
print("Feature shape:", feature_array.shape)
