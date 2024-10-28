# Heart Disease Prediction

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Summary](#data-summary)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
The Heart Disease Prediction project aims to develop a machine learning model to predict the likelihood of heart disease in individuals based on various health metrics. This project leverages historical health data to identify patterns and factors contributing to heart disease, providing valuable insights for preventive healthcare.

## Dataset
The dataset used in this project is sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and contains health records of individuals with features that indicate their health status and lifestyle choices.

## Data Summary
The dataset includes the following features:

- **HeartDiseaseorAttack**: Indicates if the individual has heart disease or has had a heart attack (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **HighBP**: Indicates if the individual has high blood pressure (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **HighChol**: Indicates if the individual has high cholesterol (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **CholCheck**: Indicates if the individual has had their cholesterol checked (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **BMI**: Body Mass Index, a measure of body fat based on height and weight.  
  - **datatype**: Numerical Continuous

- **Smoker**: Indicates if the individual is a smoker (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **Stroke**: Indicates if the individual has had a stroke (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **Diabetes**: Indicates if the individual has diabetes (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **PhysActivity**: Indicates if the individual engages in physical activity (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **Fruits**: Indicates if the individual consumes fruits (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **Veggies**: Indicates if the individual consumes vegetables (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **HvyAlcoholConsump**: Indicates if the individual consumes alcohol heavily (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **AnyHealthcare**: Indicates if the individual has any form of healthcare coverage (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **NoDocbcCost**: Indicates if the individual has had no doctor due to cost (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **GenHlth**: General health rating (1-5 scale, where 1 = poor and 5 = excellent).  
  - **datatype**: Categorical Ordinal

- **MentHlth**: Number of days in the past month when mental health was not good.  
  - **datatype**: Numerical Continuous

- **PhysHlth**: Number of days in the past month when physical health was not good.  
  - **datatype**: Numerical Continuous

- **DiffWalk**: Indicates if the individual has difficulty walking (1 = Yes, 0 = No).  
  - **datatype**: Categorical (Binary)

- **Sex**: Gender of the individual (1 = Male, 0 = Female).  
  - **datatype**: Categorical (Binary)

- **Age**: Age follows scale (1 - 18).  
  - **datatype**: Categorical Ordinal

- **Education**: Education level (scale from lower to higher education (1-5)).  
  - **datatype**: Categorical Ordinal

- **Income**: Income level (scale from low to high income categories (1-5)).  
  - **datatype**: Categorical Ordinal

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-Learn
- Imbalanced-learn (imblearn)
- XGBoost

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
git

2. Install the required packages:


```bash
pip install -r requirements.txt
```

# Usage
To train the model, run the following command:

```bash
python train_model.py
```

To make predictions, use:

```bash
python predict.py --input <data.csv>
```

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Prediction functionality
- Visualizations for data insights

## Model Training
The project utilizes various machine learning algorithms, including:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting
- XGBoost

Each model's performance is evaluated using metrics such as accuracy, precision, recall, and F1 score.

## Results
The models' results are presented in the `results` directory, including performance metrics and visualizations. The best-performing model can be found in the reports folder.

## Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature
   ```
3. Commit  changes:
   ```bash
   git commit -m 'Add  change name'
   ```
4. Push to the branch:
   ```bash
   git push origin main
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.