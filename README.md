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
