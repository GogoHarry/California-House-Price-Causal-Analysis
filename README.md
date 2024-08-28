# California-House-Price-Prediction
![house](https://github.com/user-attachments/assets/5fd68b6f-997c-4a61-a134-9bc469586716)

## Project Overview
The Californian House Price Prediction project aims to build a regression model that accurately predicts house prices in California based on various demographic and geographical features. The analysis explores how factors such as median income, house age, and proximity to the ocean influence house prices across different regions in California.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Data Exploration and Cleaning](#data-exploration-and-cleaning)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

## Dataset Description
The dataset used in this project contains information about various block groups in California, including geographical, demographic, and housing-related features. Below is a description of each feature in the dataset:
- longitude: Geographical longitude of the block group.
- latitude: Geographical latitude of the block group.
- housing_median_age: Median age of the houses within the block group.
- total_rooms: Total number of rooms in all houses within the block group.
- total_bedrooms: Total number of bedrooms in all houses within the block group.
- population: Total population of the block group.
- households: Total number of households (i.e., distinct residential units) within the block group.
- median_income: Median income for households within the block group, measured in tens of thousands of dollars.
- median_house_value: Median house value within the block group, measured in dollars.
- ocean_proximity: Proximity of the block group to the ocean (e.g., "NEAR BAY", "NEAR OCEAN").

## Installation
To run this project locally, ensure you have the following dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels joblib
```
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/GogoHarry/California-House-Price-Prediction.git
cd California-House-Price-Prediction
```
## Data Exploration and Cleaning
Initial data exploration involves inspecting the dataset for missing values, outliers, and other inconsistencies. The following steps were performed:

- Inspection of Data Types: Checking the types of variables to ensure compatibility for analysis.
- Handling Missing Values: Exploring missing data and applying appropriate strategies to handle them.
- Checking and removing outliers in median_income	and median_house_value
- Descriptive Statistics: Generating summary statistics to understand the distribution of the data.
- String Data Categorization to Dummy Variables

## Model Building
1. **Ordinary Least Squares (OLS) Regression**

The primary model used in this analysis is an OLS Regression model. Key steps in the OLS model-building process include:

1. **Adding a Constant to the Predictors:**
   - A constant is added to the predictors in both the training and test sets because statsmodels' OLS does not include it by default. 
```python
# Adding a constant to the predictors because statsmodels' OLS doesn't include it by default
X_train_const = sm.add_constant(X_train)
```
2. **Model Training:**
   - Fitting the OLS model using the training data with the added constant.
```python
Fit the OLS model
model_fitted = sm.OLS(y_train, X_train_const).fit()
```
3. **Model Summary:**
   - The model summary is printed to display key statistics and diagnostics.
```python
# Printing Summary
print(model_fitted.summary())
```
![image](https://github.com/user-attachments/assets/146a5c37-4932-4417-8518-d2eab3202463)

4. **Test Set Predictions:**
   - Predictions are made on the test set using the trained model after adding the constant to the test predictors.
```python
# Adding a constant to the test predictors
X_test_const = sm.add_constant(X_test)

# Making predictions on the test set
test_predictions = model_fitted.predict(X_test_const)
test_predictions
```
2. **Linear Regression Model (with Scaling)**

After the OLS regression, a Linear Regression model was applied with the following steps:

1. **Standardization:**
   - A StandardScaler was initialized, and the training data was scaled.
   - The same transformation was applied to the test data.
```python
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the test data
X_test_scaled = scaler.transform(X_test)
```
2. **Model Training:**
   - The Linear Regression model was created and fitted on the scaled training data.
```python
# Create and fit the model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
```
3. **Test Set Predictions:**
   - Predictions were made on the scaled test data using the trained Linear Regression model.
```python
# Make predictions on the scaled test data
y_pred = lr.predict(X_test_scaled)
```
4. **Model Evaluation:**

Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) were calculated to assess the model's performance.
```python
# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

# Output the performance metrics
print(f'MSE on Test Set: {mse}')
print(f'RMSE on Test Set: {rmse}')
```
## Model Evaluation
The model's performance was evaluated using the following metrics:
- R-squared: Measures the proportion of variance explained by the model.
- Adjusted R-squared: Adjusted version of R-squared that accounts for the number of predictors.
- F-statistic: Tests the overall significance of the model.
- Mean Squared Error (MSE): Indicates the average squared difference between observed and predicted values.
- Residual analysis and multicollinearity diagnostics were also conducted to validate the model's assumptions and stability.

## Results

The results of the Linear Regression model applied after the OLS steps are as follows:

- **Mean Squared Error (MSE) on Test Set:** 3,529,059,611.57
- **Root Mean Squared Error (RMSE) on Test Set:** 59,405.89

- **Mean Squared Error (MSE):** The MSE represents the average squared difference between the actual house prices and the predicted house prices on the test set. A higher MSE value indicates that the model's predictions deviate significantly from the actual values, which in this case is approximately 3.53 billion.

- **Root Mean Squared Error (RMSE):** The RMSE, which is the square root of the MSE, provides an error metric that is on the same scale as the target variable (house prices). An RMSE of approximately $59,405.89 suggests that, on average, the model's predictions are off by around $59,405 when predicting the house prices in the test set.

These error values indicate the typical magnitude of prediction errors. Given the scale of house prices in the dataset, this error might be significant, implying that there could be room for model improvement or that the variability in house prices is challenging to capture with the current feature set and model.

## Conclusion
The final model successfully predicts Californian house prices based on the selected features, explaining a significant portion of the variance in house prices. The insights from this model can be used to understand the factors driving property values and potentially guide investment and policy decisions in the real estate sector.

## License
This project is licensed under the MIT License - See the [LICENSE](LICENSE) file for details.

