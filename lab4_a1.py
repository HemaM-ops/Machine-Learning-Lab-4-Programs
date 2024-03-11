import pandas as pd
import numpy as np

# Load labels dataset
labels_file_path = "C:\\Users\\Anu\\Documents\\SEM_04\\ML\\PROJECT\\Legality prediction\\label.xlsx"
labels_df = pd.read_excel(labels_file_path)

# Extract labels starting from the second row of the first column
labels = labels_df.iloc[1:, 0].values.astype(int)

#  the mean (centroid) for each class
class_means = []
for class_label in np.unique(labels):
    class_indices = np.where(labels == class_label)[0]
    class_mean = np.mean(class_indices)
    class_means.append(class_mean)

#  spread (standard deviation) for each class
class_spreads = []
for class_label in np.unique(labels):
    class_indices = np.where(labels == class_label)[0]
    class_spread = np.std(class_indices)
    class_spreads.append(class_spread)

# two classes for interclass distance calculation
class_label_1 = np.unique(labels)[0]  # Class label 1
class_label_2 = np.unique(labels)[1]  # Class label 2
class_indices_1 = np.where(labels == class_label_1)[0]
class_indices_2 = np.where(labels == class_label_2)[0]
interclass_distance = np.linalg.norm(np.mean(class_indices_1) - np.mean(class_indices_2))

# Print results
print("Class Means (Centroids):", class_means)
print("Class Spreads (Standard Deviations):", class_spreads)
print("Interclass Distance between Class", class_label_1, "and Class", class_label_2, ":", interclass_distance)
