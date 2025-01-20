# Uber Fare Prediction Project

## Overview
This project aims to predict the fare amount for Uber rides based on features like pickup and drop-off locations, distance, and time of the ride. The project involves data preprocessing, feature engineering, and building a predictive machine learning model.

## Dataset
The dataset used in this project contains information about Uber rides, including:
- Pickup and drop-off coordinates (latitude and longitude)
- Fare amount
- Pickup datetime

## Steps and Methodology

### 1. Data Loading
The dataset is loaded into a pandas DataFrame:
```python
import pandas as pd

data = pd.read_csv("path_to_dataset.csv")
```

### 2. Data Cleaning
- Converted the `pickup_datetime` column to a datetime object.
- Removed rows with missing or invalid data.
- Filtered out rides with unrealistic fare amounts (e.g., below $0 or above $100).
- Ensured geographic coordinates fall within valid ranges:
  - Longitude: [-180, 180]
  - Latitude: [-90, 90]

### 3. Feature Engineering
Several new features were created to enhance the predictive power of the model:
- **Distance**: Calculated using the geodesic distance between pickup and drop-off points.
- **Hour of the Day**: Extracted from the `pickup_datetime` column.
- **Day of the Week**: Derived to capture weekday/weekend trends.
- **Rush Hour Indicator**: Flagged hours typically associated with higher traffic.
- **Distance Categories**: Binned distances into categories like `Very Short`, `Short`, etc.
- **Log Transformations**: Applied to skewed features (e.g., `fare_amount` and `distance`).

### 4. Handling Outliers
Outliers in `fare_amount` and `distance` were capped using the 99th percentile values.

### 5. Model Training
The following steps were followed for model training:
- Split the data into training and testing sets.
- Used a `RandomForestRegressor` for prediction.
- Evaluated the model using metrics such as:
  - Mean Squared Error (MSE)
  - R-squared (R²)

### 6. Visualization and Analysis
- **Correlation Analysis**: Heatmaps to understand feature relationships.
- **Feature Importance**: Assessed the contribution of each feature to the model's predictions.
- **Residual Analysis**: Checked the distribution of prediction errors.

## Requirements
Install the necessary Python libraries before running the code:
```bash
pip install pandas numpy geopy scikit-learn matplotlib seaborn
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/uber-fare-prediction.git
   ```
2. Navigate to the project directory and open the Jupyter Notebook.
3. Run all cells to preprocess the data, train the model, and visualize results.

## Results
The trained model achieved the following performance metrics:
- Mean Squared Error: `X.XX`
- R-squared: `X.XX`

## Future Work
- Incorporate weather data to improve fare predictions.
- Experiment with deep learning models for better accuracy.
- Deploy the model as a web application.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.
