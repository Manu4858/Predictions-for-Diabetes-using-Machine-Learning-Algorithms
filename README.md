# 📊 Diabetes Prediction Using Decision Tree Classifier
This project uses the Pima Indians Diabetes dataset to build and evaluate a Decision Tree Classification model that predicts whether a patient has diabetes based on diagnostic measurements.

## 🔍 Overview
Machine Learning Model: DecisionTreeClassifier

Dataset: diabetes.csv (Pima Indians Diabetes Dataset)

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

Visualizations:

Confusion Matrix Heatmap

Decision Tree Diagram

## 📁 Dataset Description
The dataset includes the following features:

Feature	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration
BloodPressure	Diastolic blood pressure
SkinThickness	Triceps skinfold thickness
Insulin	2-Hour serum insulin
BMI	Body Mass Index
DiabetesPedigree	Diabetes pedigree function
Age	Age in years
Outcome	1 = diabetes, 0 = no diabetes

## 🧪 Model Pipeline
Data Loading & Exploration

Preprocessing

Null value check

Feature-target split

Model Training

Train/test split (80/20)

Decision Tree model fitting

Model Evaluation

Confusion Matrix

Accuracy, Precision, Recall, F1-Score

Visualization

Confusion Matrix Heatmap

Decision Tree Plot

## 📈 Evaluation Results
After training the Decision Tree model, the following metrics were observed:

Accuracy: ~0.74 (example)

Precision: ~0.70

Recall: ~0.65

F1 Score: ~0.67

Note: These scores may vary depending on the model parameters and dataset version.

## 📊 Visualizations
✅ Confusion Matrix Heatmap
Provides a clear breakdown of correct and incorrect predictions.

## 🌳 Decision Tree Plot
Helps understand how the decision tree model splits the data.

## 🚀 Getting Started
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/diabetes-decision-tree.git
cd diabetes-decision-tree
Install required libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
Run the script:

bash
Copy
Edit
python diabetes_decision_tree.py
## 📂 Folder Structure
bash
Copy
Edit
├── diabetes.csv                 # Dataset
├── diabetes_decision_tree.py    # Main Python script
├── README.md                    # Project README
## ✍️ Author
Your Name
GitHub | LinkedIn

## 📜 License
This project is licensed under the MIT License.
