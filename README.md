# DATA-SCIENCE-TASKS
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
