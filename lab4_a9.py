import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load the dataset
file_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(file_path)

# Assuming 'X' contains your feature vectors and 'y' contains your class labels
X = data.drop(columns=['Judgement Status'])  # Assuming 'label' is the column containing class labels
y = data['Judgement Status']

# Perform the train-test split with 70% of the data for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the kNN classifier with k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

# Predictions on training and test sets
train_predictions = knn_classifier.predict(X_train)
test_predictions = knn_classifier.predict(X_test)

# Confusion matrix, precision, recall, and F1-Score for training set
train_conf_matrix = confusion_matrix(y_train, train_predictions)
train_precision = precision_score(y_train, train_predictions, average='weighted')  # Set average to 'weighted'
train_recall = recall_score(y_train, train_predictions, average='weighted')  # Set average to 'weighted'
train_f1_score = f1_score(y_train, train_predictions, average='weighted')  # Set average to 'weighted'

# Confusion matrix, precision, recall, and F1-Score for test set
test_conf_matrix = confusion_matrix(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='weighted')  # Set average to 'weighted'
test_recall = recall_score(y_test, test_predictions, average='weighted')  # Set average to 'weighted'
test_f1_score = f1_score(y_test, test_predictions, average='weighted')  # Set average to 'weighted'


# Print confusion matrix and performance metrics
print("Confusion Matrix for Training Set:")
print(train_conf_matrix)
print("Precision for Training Set:", train_precision)
print("Recall for Training Set:", train_recall)
print("F1-Score for Training Set:", train_f1_score)
print("\nConfusion Matrix for Test Set:")
print(test_conf_matrix)
print("Precision for Test Set:", test_precision)
print("Recall for Test Set:", test_recall)
print("F1-Score for Test Set:", test_f1_score)


#  If the model performs well on both training and test sets, it is regular fit.
#  If the model performs poorly on both training and test sets, it is underfit.
#  If the model performs well on the training set but poorly on the test set, it is overfit.
