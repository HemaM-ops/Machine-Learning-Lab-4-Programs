import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


file_path = "C:\\Users\\mahad\\OneDrive\\Desktop\\SEM 4 Courses\\ML\\Legality\\Datasets\\English_Abstractive_Embeddings_Fasttext.xlsx"
data = pd.read_excel(file_path)

#  'X' contains  feature vectors and 'y' contains  class labels
X = data.drop(columns=['Judgement Status'])  # Assuming 'label' is the column containing class labels
y = data['Judgement Status']

# the train-test split with 70% of the data for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  a kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)



# Training the classifier using the training set
neigh.fit(X_train, y_train)

# question 6
accuracy = neigh.score(X_test, y_test)
print("Accuracy of the kNN classifier:", accuracy)

#question 7

# Use the predict() function to predict classes for the test vectors
predictions = neigh.predict(X_test)

# Displaying the predicted class labels for the test vectors
print("Predicted class labels for the test vectors:")
print(predictions)

# question 8
nn_classifier = KNeighborsClassifier(n_neighbors=1)
nn_classifier.fit(X_train, y_train)

# Test accuracy of the NN classifier
nn_accuracy = nn_classifier.score(X_test, y_test)
print("Accuracy of NN classifier (k = 1):", nn_accuracy)

# Vary k from 1 to 11 for the kNN classifier and store accuracy values
k_values = range(1, 12)
knn_accuracies = []
for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_accuracy = knn_classifier.score(X_test, y_test)
    knn_accuracies.append(knn_accuracy)

# Plotting accuracy for both kNN and NN classifiers
plt.plot(k_values, knn_accuracies, label='kNN (k from 1 to 11)')
plt.axhline(y=nn_accuracy, color='r', linestyle='--', label='NN (k = 1)')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison between kNN and NN Classifiers')
plt.legend()
plt.show()
