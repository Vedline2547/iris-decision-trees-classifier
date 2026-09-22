# 🌸 Iris Flower Classification Using Decision Tree

This project demonstrates how to build a **Decision Tree Classification model** using the classic **Iris dataset** with Python and Scikit-learn.

The model learns from flower measurements and predicts which species of Iris a flower belongs to.

## 📌 Project Overview

The Iris dataset contains measurements of three different Iris flower species:

* 🌱 Iris Setosa
* 🌸 Iris Versicolor
* 🌺 Iris Virginica

The model uses four features to classify the flowers:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

A **Decision Tree Classifier** is trained on 80% of the dataset and evaluated on the remaining 20%.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib

## 📂 Project Structure

```text
Iris-Decision-Tree/
│
├── main.py
├── README.md
└── requirements.txt
```

## 📊 Dataset

The project uses the built-in Iris dataset provided by Scikit-learn.

The dataset contains:

* **150 samples**
* **4 input features**
* **3 target classes**

### Features

| Feature      | Description               |
| ------------ | ------------------------- |
| Sepal Length | Length of the sepal in cm |
| Sepal Width  | Width of the sepal in cm  |
| Petal Length | Length of the petal in cm |
| Petal Width  | Width of the petal in cm  |

### Target Classes

| Class | Species         |
| ----- | --------------- |
| 0     | Iris Setosa     |
| 1     | Iris Versicolor |
| 2     | Iris Virginica  |

## ⚙️ How the Model Works

The project follows these main steps:

```text
Load Iris Dataset
       ↓
Create Pandas DataFrame
       ↓
Separate Features and Target
       ↓
Split Data into Training and Testing Sets
       ↓
Train Decision Tree Classifier
       ↓
Make Predictions
       ↓
Evaluate Model
       ↓
Visualize Decision Tree
```

## 🧠 Machine Learning Model

The project uses a **Decision Tree Classifier**.

A decision tree makes predictions by learning a series of conditions based on the input features.

For example, the model may learn that certain petal measurements are useful for distinguishing between different Iris species.

```python
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
```

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of test samples that were classified correctly.

```python
accuracy = metrics.accuracy_score(y_test, y_pred)
```

### Confusion Matrix

Shows the number of correct and incorrect predictions for each Iris species.

```python
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
```

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

```python
class_report = metrics.classification_report(y_test, y_pred)
```

## 🌳 Decision Tree Visualization

The trained Decision Tree is visualized using Matplotlib and Scikit-learn.

```python
plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
```

This makes it easier to understand how the model makes its classification decisions.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Vedline2547/iris-Decision-Tree.git
```

Navigate into the project directory:

```bash
cd Iris-Decision-Tree
```

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib
```

## ▶️ Running the Project

Run the Python script:

```bash
python main.py
```

The program will:

1. Display the first few rows of the Iris dataset.
2. Train the Decision Tree model.
3. Predict the test data.
4. Display the model accuracy.
5. Display the confusion matrix.
6. Display the classification report.
7. Visualize the trained Decision Tree.

## 📚 What I Learned

Through this project, I practiced:

* Loading datasets using Scikit-learn
* Creating DataFrames with Pandas
* Separating features and targets
* Splitting datasets into training and testing sets
* Training a Decision Tree Classifier
* Making predictions
* Evaluating classification models
* Understanding confusion matrices
* Interpreting classification reports
* Visualizing machine learning models

## 🔮 Future Improvements

Possible improvements include:

* Experimenting with different tree depths
* Comparing Decision Trees with Random Forest
* Using cross-validation
* Tuning hyperparameters with GridSearchCV
* Comparing multiple classification algorithms
* Adding feature importance visualization
* Creating an interactive prediction interface

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning & AI Enthusiast | Python Developer

Interested in combining **Artificial Intelligence, Machine Learning, and Engineering** to develop data-driven solutions.

## ⭐ Acknowledgements

This project uses the Iris dataset provided by **Scikit-learn**.

If you found this project useful, feel free to ⭐ the repository.
