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
### 5. Feature Engineering and Data Processing
### 6. Model Evaluation

