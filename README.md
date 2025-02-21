# IPL-Score-Prediction-using-Machine-Learning

Overview

This project builds a machine learning model to predict the final score of an IPL match based on historical match data. The dataset is processed using Pandas, visualized with Seaborn, and models are trained using Scikit-learn.

Features

Data Cleaning and Preprocessing

One-Hot Encoding for Categorical Data

Train-Test Split

Multiple Regression Models:

Linear Regression

Decision Tree Regression

Random Forest Regression

AdaBoost Regression

Model Evaluation using MAE, MSE, and RMSE

Score Prediction for Ongoing Matches

Installation

Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

pip install pandas numpy seaborn matplotlib scikit-learn

Usage

1. Load the Dataset

Make sure the dataset ipl.csv is present in the working directory.

import pandas as pd
import numpy as np
df = pd.read_csv('ipl.csv')

2. Run the Model Training Script

Execute the Python script to train the models:

python model_training.py

3. Predict IPL Scores

Use the predict_score() function to estimate match scores:

final_score = predict_score(batting_team='Mumbai Indians',
                            bowling_team='Chennai Super Kings',
                            overs=10.5, runs=67, wickets=3,
                            runs_in_prev_5=29, wickets_in_prev_5=1)
print(f"Predicted final score range: {final_score-10} to {final_score+5}")

Results

The trained models are evaluated using standard regression metrics.

Random Forest and AdaBoost performed better than Linear Regression.

Predictions can be made dynamically for any match situation.

Future Improvements

Use Deep Learning models for better accuracy.

Implement a Web Interface for real-time predictions.

Expand dataset for more seasons.

Contributing

Feel free to fork this repository, make improvements, and submit a pull request.

License

This project is licensed under the MIT License.
