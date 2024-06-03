# Predicting Car Prices Using Multiple Linear Regression

In this project, we used a dataset to predict car prices based on various features using multiple linear regression.

## Table of Contents
1. Introduction
2. Objectives
3. Dataset Information
4. Prerequisites
5. Steps Involved in data preprocesing and model development
    1. Importing Libraries
    2. Loading the Dataset
    3. Data Exploration
    4. Exploratory Data Analysis
    5. Pre-processing
    6. Splitting Data into Training and Testing Sets
    7. Building and Training the Multiple Linear Regression Model
    8. Making Predictions
    9. Evaluating Model Performance
7. Conclusion

## 1. Introduction
Multiple linear regression extends simple linear regression by allowing for multiple independent variables to predict the dependent variable. This project focuses on predicting car prices based on various attributes of the cars.

## 2. Objectives
The main objectives of this project are:
- To understand the relationship between car features and their prices.
- To build a model that can predict car prices based on these features.
- To evaluate the model's performance using various metrics.

## 3. Dataset Information
The data dictionary for the car prices dataset contains the following columns, which provide detailed information about each attribute in the dataset. Here's an overview of each column:

### Dataset Information
The dataset consists of the following columns:

1. **symboling**: Insurance risk rating assigned to each car.
2. **normalized_losses**: Normalized loss values representing the relative average loss payment per insured vehicle year.
3. **make**: Manufacturer or brand of the car.
4. **fuel_type**: Type of fuel used by the car.
5. **aspiration**: Type of aspiration used in the engine.
6. **num_doors**: Number of doors in the car.
7. **body_style**: Body style of the car.
8. **drive_wheels**: Drive wheels configuration.
9. **engine_location**: Location of the engine.
10. **wheel_base**: Distance between the centers of the front and rear wheels.
11. **length**: Overall length of the car.
12. **width**: Overall width of the car.
13. **height**: Overall height of the car.
14. **curb_weight**: Weight of the car without occupants or baggage.
15. **engine_type**: Type of engine used in the car.
16. **num_cylinders**: Number of cylinders in the engine.
17. **engine_size**: Size of the engine.
18. **fuel_system**: Fuel system used in the car.
19. **bore**: Diameter of each cylinder.
20. **stroke**: Length of the piston travel inside the cylinder.
21. **compression_ratio**: Ratio of the volume of the cylinder and combustion chamber at their maximum and minimum.
22. **horsepower**: Power output of the engine.
23. **peak_rpm**: Engine's peak revolutions per minute.
24. **city_mpg**: Fuel efficiency in city driving conditions.
25. **highway_mpg**: Fuel efficiency on the highway.
26. **price**: Price of the car (target variable).

## 4. Prerequisites
To run this project, we'll need:
- Python installed on your computer.
- Basic understanding of Python programming.
- Libraries: pandas, matplotlib, seaborn, sklearn

## 5. Steps Involved in Data Preprocessing and Model Development

### A. Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```
First, we import the necessary libraries that are essential for data processing, model training, and assessment. This includes **Pandas** for data manipulation, **Matplotlib** and **Seaborn** for visualization, and various modules from **scikit-learn** for model building and evaluation.

### B. Loading the Dataset

```python
df = pd.read_csv("CarPrice_Assignment.csv")
```
We load our dataset from a CSV file using the Pandas library. This command reads the file and stores its content in a DataFrame, which allows us to manipulate and analyze the data.

### C. Data Exploration

```python
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe().T)
```
To explore our dataset and gain insights into its structure, we:
- Print the first few rows to get an initial understanding of the data.
- Check the dimensions of the dataset (number of rows and columns).
- Identify and count any missing values.
- Generate descriptive statistics for numerical columns to understand their distribution, central tendency, and dispersion.

### D. Exploratory Data Analysis
We analyze the distribution of the target variable (price) and explore correlations among features.

#### 1. Distribution of Target Variable (Price)

```python
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], kde=True)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Car Prices")
plt.grid(True)
plt.show()
```
We create a histogram to visualize the distribution of car prices in the dataset, which helps us understand the range and frequency of prices.

#### 2. Correlation Heatmap

```python
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```
We use a heatmap to illustrate the correlation between different features in the dataset. This helps identify which features have strong relationships with the target variable (price).

### E. Pre-processing

#### 1. Handling Missing Values

```python
df = df.dropna()
```
We drop any rows with missing values to ensure the dataset is complete and ready for analysis.

#### 2. Encoding Categorical Variables

```python
df = pd.get_dummies(df, drop_first=True)
```
We convert categorical variables into numerical format using one-hot encoding, which creates binary columns for each category, excluding the first to avoid multicollinearity.

#### 3. Splitting Features and Target

```python
X = df.drop('price', axis=1)
y = df['price']
```
We separate the dataset into features (X) and the target variable (y). The features include all columns except the target column 'price'.

### F. Splitting Data into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
We partition the dataset into training and testing subsets, with 70% of the data used for training and 30% for testing. This split allows us to train the model on one set and evaluate its performance on another.

### G. Building and Training the Multiple Linear Regression Model

```python
LR = LinearRegression()
LR.fit(X_train, y_train)
```
We create a linear regression model and train it using the training data. The model learns the relationship between the features and the target variable.

### H. Making Predictions

```python
y_pred = LR.predict(X_test)
```
We use the trained model to predict car prices for the test set.

### I. Evaluating Model Performance
We evaluate the model using various metrics.

#### 1. Mean Squared Error (MSE)

```python
MSE = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", MSE)
```
We calculate the Mean Squared Error (MSE) to measure the average squared difference between the predicted and actual prices.

#### 2. Mean Absolute Error (MAE)

```python
MAE = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", MAE)
```
We calculate the Mean Absolute Error (MAE) to measure the average absolute difference between the predicted and actual prices.

#### 3. Root Mean Squared Error (RMSE)

```python
from math import sqrt
RMSE = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", RMSE)
```
We compute the Root Mean Squared Error (RMSE), which is the square root of the Mean Squared Error, providing error in the same units as the target variable.

#### 4. R² Score

```python
R2_score = r2_score(y_test, y_pred)
print("R² Score:", R2_score)
```
The R² score indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It measures how well the regression model fits the observed data.

## 6. Conclusion
Based on the evaluation metrics obtained for the linear regression model:

- **Mean Absolute Error (MAE):** The MAE was calculated as 2241.650719048572, representing the average absolute difference between the actual and predicted car prices. Lower values indicate better model performance.

- **Root Mean Squared Error (RMSE):** The RMSE, calculated as 56.090520980769405, measures the average magnitude of errors between the actual and predicted car prices. It is interpreted in the same unit as the target variable and provides a sense of the model's prediction accuracy.

- **R-squared (R²) Score:** The R² score, obtained as 0.8586553507420542, reflects the proportion of variance in the target variable (car prices) explained by the model. A value close to 1 indicates a good fit of the model to the data, with higher values representing better performance.

 - **Training Accuracy:** The model achieved a score of 0.8741926830108556, indicating how well it performed on the training dataset. This metric assesses the model's ability to fit the training data.

- **Testing Accuracy:** The model's score on the testing dataset was 0.8586553507420542. This metric gauges how well the model generalizes to new, unseen data, providing insight into its predictive performance.

Considering these evaluation metrics, the linear regression model demonstrates promising predictive performance:

- The model achieved high accuracy on both the training and testing datasets, indicating robustness and generalization capability.
- The relatively low MAE and RMSE suggest that the model's predictions are close to the actual car prices on average.
- The R² score of 0.8586553507420542 indicates that the model explains approximately 85% of the variance in car prices, signifying a strong fit to the data.

Overall, based on these metrics, the linear regression model appears to be effective in predicting car prices based on the selected features. However, further analysis and comparison with alternative models could provide additional insights into its performance and potential areas for improvement.

## Important Points to Remember
- Multiple linear regression assumes a linear relationship between the independent and dependent variables.
- The model's performance can be evaluated using metrics such as MSE, MAE, RMSE, and R² Score.
- Understanding the data and checking for assumptions (linearity, independence, homoscedasticity, and normality) is crucial for building a reliable model.

This project serves as a practical introduction to multiple linear regression, showcasing how machine learning can be used to make predictions based on data.
