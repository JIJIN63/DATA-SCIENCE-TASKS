# DATA-SCIENCE-TASKS-1
# Create a bar chart or histogram to visualize the distribution of a categorical or continuous variable, such as the distribution of ages or genders in a population.
import matplotlib.pyplot as plt
ages = [25, 30, 22, 35, 28, 40, 21, 26, 32, 29, 24, 31, 27, 36, 23, 38, 30, 33, 22, 34]
plt.figure(figsize=(8, 5))
plt.hist(ages, bins=6, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

OUTPUT:
![image](https://github.com/user-attachments/assets/f0525407-2319-4cff-b71b-485fa20e4b78)

# DATA-SCIENCE-TASKS-2
# Perform data cleaning and exploratory data analysis (EDA) on a dataset of your choice, such as the Titanic dataset from Kaggle. Explore the relationships between variables and identify patterns and trends in the data.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
titanic = sns.load_dataset('titanic')
titanic_cleaned = titanic.drop(columns=['deck', 'embark_town'])
titanic_cleaned['age'] = titanic_cleaned['age'].fillna(titanic_cleaned['age'].median())
titanic_cleaned['embarked'] = titanic_cleaned['embarked'].fillna(titanic_cleaned['embarked'].mode()[0])
titanic_cleaned.dropna(subset=['embarked', 'age'], inplace=True)
survival_by_gender = titanic_cleaned.groupby('sex')['survived'].mean()
print("\nSurvival Rate by Gender:\n", survival_by_gender)
survival_by_class = titanic_cleaned.groupby('pclass')['survived'].mean()
print("\nSurvival Rate by Passenger Class:\n", survival_by_class)
plt.figure(figsize=(10, 5))
sns.histplot(data=titanic_cleaned, x='age', hue='survived', multiple='stack', bins=30, palette='Set2')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(titanic_cleaned.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

OUTPUT:
Survival Rate by Gender:
 sex
female    0.742038
male      0.188908
Name: survived, dtype: float64

Survival Rate by Passenger Class:
 pclass
1    0.629630
2    0.472826
3    0.242363
Name: survived, dtype: float64
![image](https://github.com/user-attachments/assets/9352bdd4-d00c-4213-98c9-c9ae00f157ef)
![image](https://github.com/user-attachments/assets/6fe9957a-e5a1-4e34-853f-4953e33bafe7)

# DATA-SCIENCE-TASKS-3
# Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
try:
    import requests
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for bad status codes
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Assuming 'bank-additional-full.csv' is the correct file name within the zip
        with z.open('bank-additional/bank-additional-full.csv') as f:
            df = pd.read_csv(f, sep=';')
except requests.exceptions.RequestException as e:
    print(f"Error downloading the file: {e}")
    exit()
except KeyError:
    print("Error: 'bank-additional/bank-additional-full.csv' not found in the zip file.")
    print("Available files in the zip:")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.printdir()
    exit()
print("Dataset Loaded:")
print(df.head())
df_encoded = df.copy()
label_encoders = {}
for column in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df_encoded[column])
    label_encoders[column] = le
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['no', 'yes'])
plt.title("Decision Tree (max_depth=5)")
plt.show()
plt.figure(figsize=(8, 6))
feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', color='teal')
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


OUTPUT: 
Dataset Loaded:
   age        job  marital    education  default housing loan    contact  \
0   56  housemaid  married     basic.4y       no      no   no  telephone   
1   57   services  married  high.school  unknown      no   no  telephone   
2   37   services  married  high.school       no     yes   no  telephone   
3   40     admin.  married     basic.6y       no      no   no  telephone   
4   56   services  married  high.school       no      no  yes  telephone   

  month day_of_week  ...  campaign  pdays  previous     poutcome emp.var.rate  \
0   may         mon  ...         1    999         0  nonexistent          1.1   
1   may         mon  ...         1    999         0  nonexistent          1.1   
2   may         mon  ...         1    999         0  nonexistent          1.1   
3   may         mon  ...         1    999         0  nonexistent          1.1   
4   may         mon  ...         1    999         0  nonexistent          1.1   

   cons.price.idx  cons.conf.idx  euribor3m  nr.employed   y  
0          93.994          -36.4      4.857       5191.0  no  
1          93.994          -36.4      4.857       5191.0  no  
2          93.994          -36.4      4.857       5191.0  no  
3          93.994          -36.4      4.857       5191.0  no  
4          93.994          -36.4      4.857       5191.0  no  

[5 rows x 21 columns]

Accuracy: 0.9163227320547058

Classification Report:
               precision    recall  f1-score   support

           0       0.94      0.97      0.95     10968
           1       0.66      0.52      0.58      1389

    accuracy                           0.92     12357
   macro avg       0.80      0.75      0.77     12357
weighted avg       0.91      0.92      0.91     12357
![image](https://github.com/user-attachments/assets/b572a156-7899-4ca5-a83a-c06607270fb8)
![image](https://github.com/user-attachments/assets/dce12577-03a7-4d77-b7ed-99f432a6e492)
![image](https://github.com/user-attachments/assets/3a01f9ed-6709-4309-9689-09d16033fe75)


# DATA-SCIENCE-TASKS-4
# Analyze traffic accident data to identify patterns related to road conditions, weather, and time of day. Visualize accident hotspots and contributing factors.




