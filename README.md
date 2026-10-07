# Heavy Equipment Selling Price Prediction

Regression project for predicting the selling price of used heavy equipment from machine specifications, usage information, configuration details, and sale-related features.

Built as part of the **IIT Madras BS Degree in Data Science and Applications — Machine Learning Practice (MLP)** project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Student Information](#student-information)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Model Comparison](#model-comparison)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Final Model](#final-model)
- [Project Workflow](#project-workflow)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Key Learnings](#key-learnings)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [References](#references)

---

## Project Overview

The **Heavy Equipment Selling Price Prediction** project is a supervised regression problem based on used heavy-machinery sales data.

The objective is to predict `TargetValue`, the selling price of a machine, using information describing the machine's specifications, configuration, usage history, age, and transaction details.

The competition uses **RMSLE (Root Mean Squared Logarithmic Error)** as the evaluation metric.

Because the target is right-skewed and the competition metric operates in logarithmic space, the project models:

```python
y_log = np.log1p(y)
```

and converts predictions back to the original price scale using:

```python
pred = np.expm1(pred_log)
```

The project follows a complete machine-learning workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Train / Validation Split
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Baseline Model Comparison
   ↓
Feature Importance Analysis
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Test Prediction
   ↓
Submission CSV
```

---

## Student Information

| Field | Details |
| :--- | :--- |
| Name | Mohit Malaviya |
| Roll Number | 21F1004027 |
| Course | Machine Learning Practice |
| Institution | IIT Madras BS Degree in Data Science and Applications |

---

## Project Objectives

The main objectives of the project were:

- Perform structured exploratory data analysis on a real-world tabular dataset.
- Understand missing-value patterns and categorical feature cardinality.
- Analyse the distribution of the target variable.
- Engineer meaningful features from dates, machine age, specifications, and categorical identities.
- Build a leakage-safe preprocessing pipeline.
- Compare multiple regression models.
- Evaluate models using RMSLE.
- Use cross-validation and hyperparameter search to improve the strongest model.
- Analyse model feature importance.
- Generate predictions for the competition test set.
- Produce a final submission CSV.

---

## Dataset

The project uses the **Heavy Equipment Selling Price Prediction** competition dataset.

The supplied training data contains:

- **138,701 training rows**
- **15,000 test rows**
- `TargetValue` as the regression target

The raw training data contains machine, specification, usage, categorical, and transaction-related information.

The dataset contains substantial missingness. **35 features contain missing values**, while 15 features are complete.

Some engineering/specification fields have particularly high missingness, making missing-data handling an important part of the modelling pipeline.

### Target Variable

The target is:

```text
TargetValue
```

Target statistics observed during EDA included:

| Statistic | Value |
| :--- | ---: |
| Count | 138,701 |
| Mean | 41,521.87 |
| Median | 35,000 |
| Minimum | 7,500 |
| Maximum | 142,000 |
| Standard deviation | 26,361.13 |
| Skewness | ~0.97 |
| Kurtosis | ~0.424 |

The target is positively skewed, so the model was trained using the logarithmically transformed target.

---

## Exploratory Data Analysis

EDA was used to understand the dataset and guide modelling decisions rather than simply visualising the data.

### Target Distribution

The selling-price distribution is right-skewed.

Applying `log1p` substantially reduces the skew:

```text
Raw target skewness       ≈ 0.97
log1p(target) skewness    ≈ -0.20
```

This transformation is particularly appropriate because RMSLE evaluates errors in logarithmic space.

### Missing Values

Missing-value analysis showed that a large portion of the dataset contains incomplete specification and equipment information.

Examples include:

- `OperationalHoursMeter`
- `UtilizationTier`
- Various machine specification fields

The missingness pattern was therefore treated as an important preprocessing consideration.

### Categorical Features

The dataset contains categorical features with varying cardinalities.

High-cardinality variables were not one-hot encoded indiscriminately because this can produce a very large feature matrix.

Instead, categorical variables were separated into groups and encoded according to their characteristics.

### Machine Age

Manufacture year contained anomalous placeholder values, including values around `1000` and `1001`.

These values were treated as invalid before calculating machine age.

The engineered age feature therefore used cleaned manufacture-year information rather than blindly interpreting placeholder values as genuine historical dates.

---

## Feature Engineering

Feature engineering was performed **after the train/validation split** to reduce the risk of information leakage.

The project engineered features from dates, text-like descriptors, missingness, machine age, usage, and categorical frequency.

### Date Features

Transaction date information was decomposed into:

- `TransactionYear`
- `TransactionMonth`

### Descriptor Features

Text-like specification fields were converted into simple numerical signals, including:

- `DescriptorLength`
- `DescriptorWordCount`

### Missingness Features

Missing information itself can carry signal in equipment datasets.

Engineered missingness features included:

- `HasOperationalHours`
- `HasVariantModifier`
- `MissingFeatureCount`
- `MissingSpecCount`

### Machine Age and Usage

Additional features included:

- `AssetAge`
- `HoursPerYear`

Invalid manufacture years were converted to missing values before age calculation, and negative calculated ages were clipped to zero.

### Frequency Features

Frequency encoding was used for several high-cardinality categorical identifiers, including:

- `ProductConfigFrequency`
- `BaseClassFrequency`
- `SubClassFrequency`
- `ReleaseSeriesFrequency`
- `AssetFrequency`
- `ProductConfigYearCount`
- `ProductConfigRegionCount`

These features provide information about how frequently a particular machine configuration or grouping occurs without creating thousands of one-hot columns.

### Other Engineered Features

The project also included:

- `PremiumCabin`

The final notebook contains the complete feature-engineering implementation and experimental feature ideas.

---

## Preprocessing

The data was split into training and validation sets using:

```python
train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)
```

This produced:

- **110,960 training rows**
- **27,741 validation rows**

### Numerical Features

Numerical features were imputed using median imputation:

```python
SimpleImputer(strategy="median")
```

Median imputation is robust to outliers and skewed numerical distributions.

### Categorical Features

Categorical variables were divided into groups based on their characteristics.

The preprocessing pipeline used:

- Most-frequent imputation
- `OneHotEncoder(handle_unknown="ignore")` for low-cardinality categorical variables
- `OrdinalEncoder` for higher-cardinality categorical variables
- An ordinal encoder configuration capable of handling categories that appear during transformation but were not observed during fitting

### Linear Preprocessing

A separate numerical pipeline included:

```text
Median Imputation
      ↓
StandardScaler
```

This preprocessing was used for the linear-model path.

### Tree-Based Preprocessing

Tree models do not require feature scaling, so scaling was not required for their numerical inputs.

The transformed tree-model feature matrix contained approximately **731 features** after preprocessing.

### Leakage Control

Preprocessing objects were fitted only on the training split:

```text
Training data
    ↓
fit_transform()
```

Validation data was processed using:

```text
Validation data
    ↓
transform()
```

This ensures that imputation statistics and learned categorical mappings do not use information from the held-out validation set.

---

## Models

Several regression models were compared using the same train/validation split.

### 1. Linear Regression

Linear Regression was used as a simple baseline.

**Validation RMSLE:** approximately `0.32592`

### 2. Decision Tree

A Decision Tree provides a simple nonlinear baseline capable of modelling feature interactions.

**Validation RMSLE:** approximately `0.30014`

### 3. Random Forest

Random Forest combines many decision trees. Each tree is trained using bootstrap sampling, while a random subset of features is considered at each split.

For regression, predictions from the individual trees are averaged.

**Validation RMSLE:** approximately `0.22206`

Random Forest produced the strongest initial baseline.

### 4. LightGBM

LightGBM was evaluated as a gradient-boosting model.

It was selected for further development because it achieved competitive baseline performance while providing faster experimentation than Random Forest.

**Baseline validation RMSLE:** approximately `0.24626`

After tuning and manual refinement, LightGBM reached approximately:

**`0.20050` validation RMSLE**

### 5. XGBoost

XGBoost was evaluated as another gradient-boosting approach.

**Validation RMSLE:** approximately `0.24955`

### 6. CatBoost

CatBoost was evaluated because the dataset contains many categorical variables and CatBoost provides native categorical-feature handling.

**Validation RMSLE:** approximately `0.25629`

---

## Model Comparison

| Model | Validation RMSLE |
| :--- | ---: |
| Linear Regression | ~0.32592 |
| Decision Tree | ~0.30014 |
| Random Forest | ~0.22206 |
| LightGBM | ~0.24626 |
| XGBoost | ~0.24955 |
| CatBoost | ~0.25629 |

Random Forest produced the strongest initial baseline.

LightGBM was selected for deeper experimentation because it provided a strong combination of predictive performance and training speed, making iterative tuning more practical.

---

## Hyperparameter Tuning

A custom RMSLE scorer was used for LightGBM because the model was trained on the logarithmically transformed target.

The search explored:

```python
{
    "num_leaves": [255, 319],
    "learning_rate": [0.03],
    "n_estimators": [1000, 1200],
    "min_child_samples": [10, 20],
    "feature_fraction": [0.8],
    "bagging_fraction": [0.8],
    "lambda_l2": [2, 3]
}
```

This resulted in:

```text
16 parameter combinations
×
3-fold cross-validation
=
48 model fits
```

### Cross-Validation

`GridSearchCV` used three-fold cross-validation within the training portion of the dataset.

The separate 27,741-row validation set remained outside the GridSearch process and was used for final held-out evaluation.

The best cross-validation configuration achieved approximately:

```text
0.20785 RMSLE
```

The final manually refined LightGBM configuration achieved approximately:

```text
0.20050 validation RMSLE
```

The manually refined configuration was therefore selected based on held-out validation performance.

---

## Final Model

The final modelling workflow used the strongest LightGBM configuration developed during experimentation.

The target was modelled in log space:

```python
y_log = np.log1p(y)
```

Predictions were converted back to the original price scale using:

```python
pred = np.expm1(pred_log)
```

The final notebook then generates predictions for the competition test set and writes the submission CSV.

---

## Project Workflow

```text
Load Dataset
     ↓
Inspect Schema & Metadata
     ↓
Missing-Value Analysis
     ↓
Target Distribution Analysis
     ↓
Numerical / Categorical EDA
     ↓
Train / Validation Split
     ↓
Feature Engineering
     ↓
Preprocessing Pipelines
     ↓
Baseline Model Comparison
     ↓
LightGBM Feature Importance
     ↓
Hyperparameter Search
     ↓
Manual Refinement
     ↓
Held-Out Validation
     ↓
Train Final Model
     ↓
Predict Test Set
     ↓
submission.csv
```

---

## Repository Structure

```text
HEAVY-EQUIPMENT-SELLING-PRICE-PREDICTION/
│
├── README.md
├── notebook/
│   └── MLP_Heavy_Equipment_Selling_Price_Prediction.ipynb
│
└── submission/
    └── submission.csv
```

### Directory Description

**`notebook/`**  
Contains the complete final notebook covering EDA, feature engineering, preprocessing, model comparison, feature importance, hyperparameter tuning, and final prediction generation.

**`submission/`**  
Contains the final competition prediction CSV.

**`README.md`**  
Documents the methodology, modelling decisions, results, and repository structure.

---

## Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Kaggle

---

## Key Learnings

This project provided practical experience with:

- Exploratory data analysis on large tabular datasets
- Regression modelling
- RMSLE and logarithmic target transformation
- Missing-value analysis and imputation
- Categorical feature encoding
- High-cardinality feature handling
- Feature engineering from dates and machine metadata
- Frequency encoding
- Leakage-aware preprocessing
- Decision Trees and Random Forests
- Gradient boosting
- LightGBM
- XGBoost
- CatBoost
- Feature importance analysis
- Cross-validation
- GridSearchCV
- Model selection using a held-out validation set
- Kaggle submission workflows

A particularly important modelling lesson was that the strongest baseline is not necessarily the most practical model to develop further. LightGBM provided a useful balance between performance and experimentation speed, allowing more efficient feature engineering and hyperparameter tuning.

---

## Limitations and Future Improvements

### Frequency-map consistency

One frequency feature mapping was constructed from the broader training dataframe rather than strictly from the `X_train` split used for model fitting.

A cleaner implementation would construct every frequency map exclusively from `X_train` and then apply the mapping to validation and test data.

### Feature selection

The transformed feature matrix contains many low-importance or zero-importance features.

Future experiments could investigate systematic feature pruning and ablation studies.

### Missingness handling

The project explicitly engineered missingness-count features, while also using imputation in the shared preprocessing pipeline.

Future experiments could compare this approach with models that directly exploit native missing-value handling.

### Hyperparameter search

The LightGBM search was deliberately limited to a manageable grid.

A broader search could investigate additional combinations of tree complexity, learning rate, number of estimators, subsampling, and regularisation.

### Validation strategy

The project used a fixed random 80/20 split for model comparison.

Future work could evaluate multiple random seeds or alternative validation strategies to obtain a more robust estimate of generalisation performance.

---

## References

1. [Kaggle — Heavy Equipment Selling Price Prediction](https://www.kaggle.com/competitions/heavy-equipment-selling-price-prediction-challenge)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
4. [XGBoost Documentation](https://xgboost.readthedocs.io/)
5. [CatBoost Documentation](https://catboost.ai/docs/)
6. [Pandas Documentation](https://pandas.pydata.org/docs/)
7. [NumPy Documentation](https://numpy.org/doc/)
---

## Acknowledgements

This project was completed as part of the **Machine Learning Practice (MLP)** course in the **IIT Madras BS Degree in Data Science and Applications** programme.
