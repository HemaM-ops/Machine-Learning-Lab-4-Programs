import pandas as pd
from sklearn.model_selection import train_test_split


file_path = "C:\\Users\\Anu\\Documents\\SEM_04\\ML\\PROJECT\\Legality prediction\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(file_path)

# 'X' contains  feature vectors and 'y' contains  class labels
X = data.drop(columns=['Judgement Status'])  # Assuming 'label' is the column containing class labels
y = data['Judgement Status']

# the train-test split with 70% of the data for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)



print(X_train)

print(X_test)

print(y_train)

print(y_test)