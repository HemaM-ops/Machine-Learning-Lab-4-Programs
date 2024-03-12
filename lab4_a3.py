import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load embeddings dataset
embeddings_file_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"

embeddings_df = pd.read_excel(embeddings_file_path, header=None)

# Extract feature vectors (assuming features start from the second row)
feature_vectors = embeddings_df.iloc[1:, :].values.astype(float)

# Choose two feature vectors for distance calculation
vector1_index = 0  # Index of the first feature vector
vector2_index = 1  # Index of the second feature vector
vector1 = feature_vectors[vector1_index]
vector2 = feature_vectors[vector2_index]

# Function to calculate Minkowski distance
def minkowski_distance(x, y, r):
    return np.power(np.sum(np.abs(x - y) ** r), 1 / r)

# Calculate Minkowski distance for different values of r
r_values = range(1, 11)
distances = [minkowski_distance(vector1, vector2, r) for r in r_values]
print(distances)

# Plot the distances against r
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Distance')
plt.grid(True)
plt.show()
