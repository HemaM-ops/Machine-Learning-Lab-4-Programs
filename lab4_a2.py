import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
dataset_file_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"
dataset_df = pd.read_excel(dataset_file_path)

# Choosing a feature from your dataset
chosen_feature = dataset_df[0]

# Plotting histogram
plt.hist(chosen_feature, bins=10, color='blue', alpha=0.7)
plt.title('Histogram of Chosen Feature')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.show()

#mean and variance
mean_value = np.mean(chosen_feature)
variance_value = np.var(chosen_feature)

print("Mean:", mean_value)
print("Variance:", variance_value)
