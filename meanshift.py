import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('vital.status', axis=1)
y_true = df['vital.status']

# Function to map cluster labels to binary predictions
def map_clusters_to_binary(cluster_labels):
    # Find unique cluster labels
    unique_clusters = np.unique(cluster_labels)
    
    # For each cluster, find the majority class
    cluster_to_class = {}
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        if np.sum(mask) > 0:
            # Find majority class in this cluster
            cluster_majority = np.argmax(np.bincount(y_true[mask]))
            cluster_to_class[cluster] = cluster_majority
    
    # Map each point's cluster to its predicted class
    y_pred = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_labels):
        y_pred[i] = cluster_to_class[cluster]
    
    return y_pred

# Function to evaluate clustering results
def evaluate_clustering(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\nMean Shift Clustering Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return acc

# Apply Mean Shift clustering
print("\n--- Mean Shift Clustering ---")
# Calculate bandwidth based on the data variance
bandwidth = np.mean(np.std(X, axis=0)) * 0.5
print(f"Using bandwidth: {bandwidth:.4f}")

# Perform clustering
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_labels = meanshift.fit_predict(X)

# Analyze results
n_clusters = len(np.unique(meanshift_labels))
print(f"Number of clusters found: {n_clusters}")

# Map clusters to binary predictions and evaluate
meanshift_pred = map_clusters_to_binary(meanshift_labels)
meanshift_acc = evaluate_clustering(y_true, meanshift_pred)

# Show cluster distribution
#cluster_counts = pd.Series(meanshift_labels).value_counts().sort_index()
#print("\nCluster distribution:")
#for cluster, count in cluster_counts.items():
#    print(f"Cluster {cluster}: {count} samples")

# Print overall class distribution for comparison
class_counts = pd.Series(y_true).value_counts().sort_index()
print("\nActual class distribution:")
for label, count in class_counts.items():
    print(f"Class {label}: {count} samples")
