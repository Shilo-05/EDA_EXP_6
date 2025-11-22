# EDA_EXP_6

```
Name         : Shivaram M.
Register No. : 212223040195
```

## Aim

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

## Algorithm

1. Import pandas, numpy, seaborn, matplotlib, sklearn libraries.
2. Plot univariate distributions for alcohol, volatile acidity, and pH to understand individual feature behavior.
3. Create bivariate boxplots to study relationships between wine quality and key predictors.
4. Compute and visualize the correlation matrix to identify feature relationships with wine quality.
5. Convert wine quality into a binary good/bad label for classification.
6. Split the dataset into training and testing sets for model evaluation.
7. Train a Logistic Regression model and predict wine quality on test data.
8. Evaluate the model using accuracy and confusion matrix, and detect outliers using boxplots.

## Program

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
df=pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';')
df.head()
df.isnull().sum()
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[~((df < lower) | (df > upper)).any(axis=1)]
df = remove_outliers_iqr(df)
print("Before:", df.shape)
print("After:", df.shape)
sns.histplot(df['alcohol'],color='green',bins=30,kde=True)
sns.histplot(df['volatile acidity'],bins=30,kde=True,color='blue')
sns.histplot(df['pH'],bins=30,kde=True,color='black')
sns.lineplot(x=df['alcohol'],y=df['quality'])
sns.lineplot(x=df['fixed acidity'],y=df['quality'],color='green')
df.head()
corr=df[['fixed acidity','volatile acidity','citric acid','pH','alcohol','quality']]
sns.heatmap(corr.corr(),cbar=True,annot=True,cmap='icefire')
plt.title("HEATMAP")
Y=df[['quality']]
X=df.drop(columns='quality')
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
acc=accuracy_score(Y_test,y_pred)
print("Accuracy is : ",round(acc*100,3))
conf=confusion_matrix(Y_test,y_pred)
print(conf)
sns.heatmap(conf,cbar=True,annot=True,cmap='icefire')
plt.title("Confusion Matrix")
clas=classification_report(Y_test,y_pred,zero_division=True)
print(clas)
```

## Output
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 18 57 AM" src="https://github.com/user-attachments/assets/b0ac2f9f-efcc-431a-a987-8be0c0b9593f" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 07 AM" src="https://github.com/user-attachments/assets/4677c37a-c241-45ac-9c25-f28f87ffa947" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 12 AM" src="https://github.com/user-attachments/assets/aeeef20f-c7db-4b0e-b429-d6f0877f42f1" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 23 AM" src="https://github.com/user-attachments/assets/11d2a45c-19b0-4342-8fa7-9bd6beda4f54" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 29 AM" src="https://github.com/user-attachments/assets/6a1e3fde-6b0b-408d-a337-833ab2aa2f21" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 35 AM" src="https://github.com/user-attachments/assets/f55bd455-2c91-4305-b16b-1f0e91e32845" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 41 AM" src="https://github.com/user-attachments/assets/0b566dfe-10aa-4585-8fe9-73cc18ca378d" />
<img width="1680" height="1050" alt="Screenshot 2025-11-22 at 9 19 46 AM" src="https://github.com/user-attachments/assets/12db94ab-7e8a-459d-90ed-7b5a37370d5f" />

## Result
Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
