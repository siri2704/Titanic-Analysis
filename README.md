 Titanic-Analysis
 Titanic Survival Prediction Analysis

Project Overview
This project uses the Titanic dataset to build a predictive model for passenger survival. The model is trained using machine learning techniques and helps analyze the factors that contributed to passenger survival or death. This repository contains the Python code for the Titanic survival prediction using various features like age, sex, and class.

 Dataset
The dataset used in this project is the famous Titanic dataset, which provides data on passengers, including features like:
- Passenger ID
- Survival (1 for survived, 0 for did not survive)
- Pclass (ticket class)
- Name
- Sex
- Age
- SibSp (number of siblings/spouses aboard the Titanic)
- Parch (number of parents/children aboard the Titanic)
- Ticket number
- Fare
- Cabin number
- Embarked (port of embarkation)

The dataset is available in two parts: training and testing data. The training data is used to build the model, while the testing data evaluates its performance.

Dependencies
To run the Titanic survival analysis, the following packages need to be installed:
- Python 3.6+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Install the required dependencies using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Code Explanation

1. Data Preprocessing
The dataset is first loaded into a pandas DataFrame. Missing values, especially for 'Age' and 'Cabin', are handled by filling or removing them. Non-numeric columns like 'Sex' and 'Embarked' are encoded into numeric values for use in machine learning models.

2. Feature Selection
Important features such as 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', and 'Fare' are selected to train the model. These features are normalized where necessary to improve model performance.

3. Model Building
The model is built using Scikit-learn's machine learning algorithms. Multiple models such as Logistic Regression, Random Forest, or Decision Trees can be used to predict survival. The models are evaluated using cross-validation and performance metrics such as accuracy, precision, recall, and F1-score.

 4. Model Evaluation
The performance of the model is tested on a hold-out dataset (test set) to check its accuracy and other metrics. Confusion matrices and ROC curves can be plotted to visualize the results.

 5. Predictions
After building and testing the model, predictions are made on the test set, and the output is saved in a CSV format for submission.

How to Run the Code

1. Make sure you have installed the required dependencies:

2 Run the Titanic analysis script:

3. The results will be displayed, and the predicted outcomes can be saved for further analysis.

Output
The script will generate predictions for passenger survival based on the input test data. Additionally, it will display metrics like accuracy, precision, recall, and F1-score, helping you understand the performance of the model.


Acknowledgements
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) for providing the data used in this project.
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning tools.
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation.
