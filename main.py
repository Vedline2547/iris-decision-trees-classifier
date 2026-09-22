# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Load the dataset
iris = load_iris()
# Craete a DataFrame from the iris dataset
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
# Add the species column(target)
df['species'] = iris.target 
# Display first few rows
print("Iris Dataset: ")
print(df.head())
# Split dataset into features and targets
X = df.drop('species',axis=1)
y = df['species']
# Split dataset into training and test splits
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Train a Decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
# Make predictions on test set
y_pred = classifier.predict(X_test)
# Evaluate the model
accuracy = metrics.accuracy_score(y_test,y_pred)
conf_matrix = metrics.confusion_matrix(y_test,y_pred)
class_report = metrics.classification_report(y_test,y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print("\nConfusion matrix:")
print(conf_matrix)
print("\nClassification report:")
print(class_report)
# Visualize the Decision tree
plt.figure(figsize=(12,8))
plot_tree(classifier,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.title("Decision Tree for Iris Flower Classification")
plt.show()