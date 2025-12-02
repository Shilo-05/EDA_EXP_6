# Experiment 6: Wine Quality Analysis

**Name:** Oswald Shilo  
**Reg.No:** 212223040139  

---

## Aim
To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the Interquartile Range (IQR) method, and evaluate the performance of a Logistic Regression model in predicting wine quality.

---

## Algorithm / Procedure

1. **Import Libraries**  
   Load `pandas`, `seaborn`, `matplotlib`, and `sklearn` modules.

2. **Load Data**  
   Import the **Wine Quality Red** dataset from the UCI repository into a pandas DataFrame.

3. **Data Preprocessing**
   - Inspect for null values.  
   - Define a function `remove_outliers_iqr()` to filter out data points beyond `1.5 × IQR`.  
   - Apply outlier removal and verify dataset shape reduction.

4. **Exploratory Data Analysis (EDA)**
   - Plot histograms for features like `alcohol`, `volatile acidity`, and `pH` to visualize distributions.  
   - Use line plots to observe trends between features and `quality`.  
   - Generate a **correlation heatmap** to identify significant predictors.

5. **Model Training**
   - Define `X` (features) and `Y` (`quality`).  
   - Split the data into **training (70%)** and **testing (30%)** sets.

6. **Evaluation**
   - Train a `LogisticRegression` classifier.  
   - Predict outcomes on the test set.  
   - Calculate **accuracy** and generate a **confusion matrix** to visualize prediction errors.  
   - Print the **classification report**.

---

## Program (Python)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Load Dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')

print("First 5 rows:")
print(df.head())
print("\nNull Values:")
print(df.isnull().sum())

# 2. Outlier Removal (IQR Method)
def remove_outliers_iqr(df):
    # Only apply IQR to numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Broadcast lower/upper back to the original dataframe index
    mask = ~((numeric_df < lower) | (numeric_df > upper)).any(axis=1)
    return df[mask]

print("\nShape Before Outlier Removal:", df.shape)
df_clean = remove_outliers_iqr(df)
print("Shape After Outlier Removal:", df_clean.shape)

# 3. Visualization: Distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df_clean['alcohol'], color='green', bins=30, kde=True)
plt.title("Distribution of Alcohol")

plt.subplot(1, 3, 2)
sns.histplot(df_clean['volatile acidity'], bins=30, kde=True, color='blue')
plt.title("Distribution of Volatile Acidity")

plt.subplot(1, 3, 3)
sns.histplot(df_clean['pH'], bins=30, kde=True, color='black')
plt.title("Distribution of pH")

plt.tight_layout()
plt.show()

# 4. Visualization: Trends vs Quality
plt.figure(figsize=(10, 5))
sns.lineplot(x=df_clean['alcohol'], y=df_clean['quality'], label='Alcohol')
sns.lineplot(x=df_clean['fixed acidity'], y=df_clean['quality'], color='green', label='Fixed Acidity')
plt.title("Feature Trends vs Quality")
plt.xlabel("Feature Value")
plt.ylabel("Quality Score")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_clean[['fixed acidity',
                 'volatile acidity',
                 'citric acid',
                 'pH',
                 'alcohol',
                 'quality']]
sns.heatmap(corr.corr(), cbar=True, annot=True, cmap='icefire', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 6. Classification Model
Y = df_clean['quality']
X = df_clean.drop(columns='quality')

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# 7. Evaluation
acc = accuracy_score(Y_test, y_pred)
print("\nAccuracy is : ", round(acc * 100, 3), "%")

conf = confusion_matrix(Y_test, y_pred)
print("\nConfusion Matrix:\n", conf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf, cbar=True, annot=True, cmap='icefire', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(Y_test, y_pred, zero_division=True))
````

---

## Output

```markdown
![Figure 1 – Head, Null Check, and Shape](https://github.com/user-attachments/assets/b0ac2f9f-efcc-431a-a987-8be0c0b9593f)

![Figure 2 – Distribution of Alcohol, Volatile Acidity, pH](https://github.com/user-attachments/assets/4677c37a-c241-45ac-9c25-f28f87ffa947)

![Figure 3 – Feature Trends vs Quality](https://github.com/user-attachments/assets/aeeef20f-c7db-4b0e-b429-d6f0877f42f1)

![Figure 4 – Correlation Heatmap](https://github.com/user-attachments/assets/11d2a45c-19b0-4342-8fa7-9bd6beda4f54)

![Figure 5 – Model Training / Metrics Console](https://github.com/user-attachments/assets/6a1e3fde-6b0b-408d-a337-833ab2aa2f21)

![Figure 6 – Confusion Matrix Heatmap](https://github.com/user-attachments/assets/f55bd455-2c91-4305-b16b-1f0e91e32845)

![Figure 7 – Classification Report & Additional Text Output](https://github.com/user-attachments/assets/0b566dfe-10aa-4585-8fe9-73cc18ca378d)

![Figure 8 – Final Console Output](https://github.com/user-attachments/assets/12db94ab-7e8a-459d-90ed-7b5a37370d5f)
```

---

## Inference

* **Outlier Impact:**
  The IQR-based outlier removal significantly reduced the dataset size.
  This helps ensure that extreme values (e.g., unusually high acidity or residual sugar) do not skew the model. However, it also reduces the training data, creating a trade-off between robustness and data volume.

* **Feature Correlation:**
  The correlation heatmap shows that:

  * **Alcohol** has a strong positive correlation with **quality** (higher alcohol content often correlates with higher ratings).
  * **Volatile acidity** has a negative correlation with **quality**, as higher vinegar-like acidity generally reduces wine quality.

* **Model Performance:**
  The Logistic Regression classifier achieves **moderate accuracy**.
  The confusion matrix indicates that:

  * The model performs better at predicting **average-quality wines** (e.g., quality scores 5 or 6).
  * It struggles more with **extreme quality** classes (e.g., very low or very high scores), likely due to class imbalance.

---

## Result

The Exploratory Data Analysis on the Wine Quality dataset was successfully performed.
Outliers were detected and removed using the IQR method, and a Logistic Regression model provided a solid **baseline** for predicting wine quality based on physicochemical properties.
This experiment establishes a foundation for further improvements using more advanced models (e.g., Random Forests, Gradient Boosting) or better handling of class imbalance.

```
```
