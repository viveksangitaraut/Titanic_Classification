# Titanic_Classification
Titanic_Classification Algorithm which tells whether the person will be save from sinking or not

## Purpose
-  A famous shipwreck in history is the Titanic. 1502 out of the 2224 passengers and crew aboard the Titanic perished on April 15, 1912, when she sank after striking an iceberg during her maiden voyage. A better set of ship safety regulations resulted from this shocking tragedy that shocked the entire world.

-  Some groups of people appeared to have higher survival rates than others, despite the fact that survival sometimes involved a certain amount of luck.

-  With the help of passenger data (such as name, age, gender, socioeconomic class, etc.), you are to create a predictive model that addresses the question: "What kinds of people were more likely to survive?"

## About The Dataset

-  The data has been split into two groups:

      ### - training set (train.csv)
      ### - test set (test.csv)
-  The training set includes passengers survival status (also know as the ground truth from the titanic tragedy) which along with other features like gender, class, fare and pclass is used to create the machine learning model.

-  The test set should be used to see how well the model performs on unseen data. The test set does not provide passengers survival status. We are going to use our model to predict passenger survival status.

-  This is clearly a Classification problem. In predictive analytics, when the target is a categorical variable, we are in a category of tasks known as classification tasks.


## STEPS INVOLVED :

### 1. Problem understanding and definition
-  In order to complete this challenge, we must fully analyse the types of people who had the highest chance of surviving.

-  We specifically use machine learning tools to foretell which passengers survived the tragedy.

-  Determine the probability that the passenger survive or not.

### 2. Data Loading and Importing the necessary libraries
#### NumPy (np): 
-  It is used for performing linear algebra operations, which are fundamental in data analysis and machine learning.

#### Pandas (pd): 
-  This library is essential for data manipulation and analysis, as it provides powerful data structures like DataFrames for handling and analyzing tabular data.

#### Seaborn and Matplotlib: 
-  These libraries are used for data visualization. Seaborn provides an enhanced interface to create attractive statistical graphics, while Matplotlib is a versatile library for creating various types of plots and charts. The %matplotlib inline magic command is used to display plots within Jupyter Notebook, and style is used to customize plot styles.

#### Scikit-Learn (sklearn): 
-  I import several machine learning algorithms and tools from Scikit-Learn, including linear regression, logistic regression, random forest, perceptron, stochastic gradient descent classifier, decision tree classifier, k-nearest neighbors classifier, and Gaussian naive Bayes classifier. These algorithms will be used for building and training machine learning models later in the code.

### 3. Data understanding using Exploratory Data Analysis (EDA)
-  Exploratory Data Analysis is a crucial process that entails performing initial investigations on data in order to find patterns, identify - 
   anomalies, test hypotheses, and validate assumptions with the aid of summary statistics and graphical representations.

-  In conclusion, it's a method for analysing data sets to highlight their key features, frequently using visual techniques
  
### 4. Feature Engineering and Data Processing

#### 4.1 Drop 'PassengerId': 
The 'PassengerId' column, which doesn't contribute to survival prediction, is removed from the training dataset.

#### 4.2 Combining SibSp and Parch: 
The 'SibSp' (siblings/spouses) and 'Parch' (parents/children) columns are combined to create a new feature called 'relatives.' A 'not_alone' column is also created to indicate if a passenger is traveling alone.

#### 4.3 Missing Data Handling: 
Missing values in the 'Cabin' column are filled with "U0" and then a 'Deck' feature is created from the first letter of the cabin. Missing values in the 'Age' column are filled with random values based on the mean and standard deviation. Missing values in the 'Embarked' column are filled with the most common value, "S."

#### 4.4 Converting Features: 
The 'Fare' column is converted from float to integer data type. The 'Name' column is processed to extract titles like Mr, Miss, Mrs, Master, and Others. The 'Sex' column is mapped to numeric values (0 for male, 1 for female). The 'Ticket' column is dropped as it has too many unique values. The 'Embarked' column is mapped to numeric values (0 for S, 1 for C, 2 for Q).

#### 4.5 Creating New Categories: 
The 'Age' column is categorized into age groups. The 'Fare' column is categorized using quantiles to create fare ranges.

### 5. Feature Engineering and Data Processing
#### 5.1 Stochastic Gradient Descent (SGD): 
A linear classifier is trained using the Stochastic Gradient Descent algorithm with a maximum of 5 iterations, achieving an accuracy of approximately 79.69%.

#### 5.2 Decision Tree: 
A Decision Tree classifier is built, which partitions data into branches based on feature values, resulting in an accuracy of around 93.15%.

#### 5.3 Random Forest: 
A Random Forest classifier is constructed with 100 decision trees to improve accuracy. It achieves the same accuracy of 93.15% as the Decision Tree.

#### 5.4 Logistic Regression: 
Logistic Regression is used to model the relationship between the independent variables and binary outcome. It achieves an accuracy of approximately 81.71%.

#### 5.5 K-Nearest Neighbors (KNN): 
K-Nearest Neighbors classification with k=3 neighbors is employed, yielding an accuracy of around 86.98%.

#### 5.6 Gaussian Naive Bayes: 
The Gaussian Naive Bayes classifier is used for probabilistic classification, achieving an accuracy of approximately 78.68%.

#### 5.7 Perceptron: 
A Perceptron classifier is trained with a maximum of 1000 iterations to learn a linear decision boundary, though the accuracy score is not provided.

### 6. Model Evaluation

#### 6.1 Model Evaluation:
The code evaluates different machine learning models and their accuracies. Random Forest and Decision Tree models perform the best, with around 93% accuracy.

#### 6.2 K-Fold Cross Validation: 
Demonstrates K-Fold Cross Validation to assess the Random Forest model's performance more reliably, with an average accuracy of 81%.

#### 6.3 Feature Importance:
Identifies important features for the Random Forest model, helping to understand which attributes contribute most to predictions.

#### 6.4 Feature Selection:
Removes less significant features ('not_alone' and 'Parch') to potentially improve model performance.

#### 6.5 Hyperparameter Tuning:
Conducts a grid search to optimize Random Forest hyperparameters for better accuracy, resulting in improved performance.

#### 6.6 Testing New Parameters:
Tests the Random Forest model with the optimized hyperparameters, achieving an out-of-bag (OOB) score of around 83%.

#### 6.7 Confusion Matrix:
Computes a confusion matrix to evaluate the model's performance, showing the true positives, true negatives, false positives, and false negatives.

#### 6.8 Precision and Recall:
Calculates precision and recall scores to assess the model's ability to correctly classify survivors and non-survivors.

#### 6.9 F-score: 
Combines precision and recall into an F-score, providing a single metric for evaluating model performance (around 76%).

#### 6.10 Precision-Recall Curve:
Plots precision and recall against different thresholds, helping to choose an appropriate trade-off between the two metrics.

#### 6.11 ROC AUC Curve: 
Illustrates the Receiver Operating Characteristic (ROC) curve and area under the curve (ROC AUC Score) to assess model performance in binary classification tasks (ROC AUC Score of 93%).

## Conclusion

### - Certainly, here are the key conclusions from this Titanic classification project:

#### - The project aimed to predict survival outcomes of Titanic passengers using machine learning techniques.

#### - Exploratory Data Analysis (EDA) was performed to understand the dataset and identify patterns & relationships among variables.

#### - Feature engineering was crucial, including handling missing data, converting categorical features, and creating new meaningful attributes like 'relatives' and 'Deck.'

#### - Several classification algorithms were applied, including Stochastic Gradient Descent, Decision Trees, Random Forest, Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, and Perceptron.

#### - Decision Tree and Random Forest models achieved the highest accuracy of approximately 93%, outperforming other models.

#### - K-Fold Cross Validation was used to validate model performance more reliably, with an average accuracy of 81%.

#### - Feature importance analysis indicated that certain attributes, like 'Sex' and 'Pclass,' significantly influenced survival predictions.

#### - Hyperparameter tuning improved Random Forest's performance, reaching an OOB score of around 83%.

#### - Precision, recall, and F-score metrics were used to evaluate model performance, with F-score around 76%.

#### - ROC AUC analysis demonstrated the model's ability to distinguish between survivors and non-survivors, with an ROC AUC score of 93%.
