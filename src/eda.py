import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 1. Load Dataset ---
DATASET_PATH = 'data/raw_csv/tactical_landmarks.csv'

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}. Run collect_data.py first!")

df = pd.read_csv(DATASET_PATH)

print("=" * 50)
print("       EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# --- 2. Dataset Overview ---
print(f"\n[1] Total Samples Recorded: {len(df)}")
print(f"[2] Total Feature Columns: {len(df.columns) - 1} (63 features: 21 joints x [X, Y, Z])")

# --- 3. Class Balance Check ---
print("\n[3] Class Distribution:")
class_counts = df['label'].value_counts()
print(class_counts.to_string())

if class_counts.min() / class_counts.max() < 0.7:
    print("\n[WARNING] Imbalanced classes detected! Try recording more samples for underrepresented gestures.")
else:
    print("\n[SUCCESS] Class distribution is well-balanced.")

# --- 4. Outlier & Range Check ---
features = df.drop(columns=['label'])
labels = df['label']

# Check for NaN values
nan_count = features.isna().sum().sum()
print(f"\n[4] Missing / NaN Values: {nan_count}")

# Check magnitude range ( wrist relative coords should be roughly in [-1.5, 1.5] )
max_val = features.abs().max().max()
print(f"[5] Maximum Relative Landmark Magnitude: {max_val:.4f}")
if max_val > 2.0:
    print("   [NOTE] Some extreme coordinate values detected (possible MediaPipe misdetection).")

# --- 5. Visualizing Feature Separation via PCA (2D Plot) ---
print("\n[6] Generating PCA Cluster Visualization...")

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

plt.figure(figsize=(10, 6))
unique_labels = labels.unique()

for label in unique_labels:
    idx = labels == label
    plt.scatter(
        features_2d[idx, 0], 
        features_2d[idx, 1], 
        label=label, 
        alpha=0.7, 
        edgecolors='k', 
        s=40
    )

plt.title("Tactical Gestures - 2D PCA Feature Separation", fontsize=14, fontweight='bold')
plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
plt.legend(title="Gestures")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save PCA plot
os.makedirs('data/processed', exist_ok=True)
plt.savefig('data/processed/pca_clusters.png', dpi=300)
print("   Saved plot to: data/processed/pca_clusters.png")

plt.show()