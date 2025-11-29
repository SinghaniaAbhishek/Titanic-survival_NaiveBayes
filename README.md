# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using the Naive Bayes classification algorithm.

## Overview

This project implements a **Gaussian Naive Bayes** classifier to predict whether a Titanic passenger would have survived based on their demographic and ticket information. The model analyzes historical passenger data and makes binary predictions (survived/not survived).

## Project Structure

```
Titanic-survival_NaiveBayes/
├── titanic_survival.py          # Standalone Python script
├── TITANIC_SURVIVAL.ipynb       # Jupyter notebook with detailed analysis
├── titanicsurvival.csv          # Dataset file
└── README.md                    # Project documentation
```

## Dataset

**File:** `titanicsurvival.csv`

**Features used:**
- `Pclass` - Passenger class (1, 2, or 3)
- `Sex` - Passenger gender (encoded: 0=female, 1=male)
- `Age` - Passenger age in years
- `Fare` - Ticket fare amount

**Target:** `Survived` - Binary outcome (0=did not survive, 1=survived)

## Methodology

### Data Preprocessing
1. **Encoding:** Convert categorical gender values (female→0, male→1)
2. **Handling Missing Values:** Fill missing age values with mean age
3. **Feature Extraction:** Separate features (X) from target variable (Y)

### Model Training
- **Algorithm:** Gaussian Naive Bayes
- **Train-Test Split:** 75% training, 25% testing
- **Performance Metric:** Accuracy score

### Predictions
The model accepts user input for:
- Passenger class
- Gender (0 for female, 1 for male)
- Age
- Fare

## Requirements

```
pandas
numpy
scikit-learn
```

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

### Running the Python Script

```bash
python titanic_survival.py
```

When prompted, enter passenger information:
```
Enter Person's Pclass number: 1
Enter Person's Gender 0-female 1-male(0 or 1): 0
Enter Person's Age: 25
Enter Person's Fare: 100.0
```

### Running the Jupyter Notebook

```bash
jupyter notebook TITANIC_SURVIVAL.ipynb
```

## Results

The model outputs:
- Prediction for individual passengers
- Comparison of predicted vs. actual test results
- Overall model accuracy percentage

## Key Insights

- The Naive Bayes algorithm provides a baseline classification model
- Feature importance varies: passenger class and gender are strong predictors
- Model handles missing data through mean imputation
- Accuracy metric validates model performance on test data

## Author

SinghaniaAbhishek

## License

MIT License

---

**Note:** This is a classification model for educational purposes. Predictions are probabilistic estimates based on historical data patterns, not guarantees.
