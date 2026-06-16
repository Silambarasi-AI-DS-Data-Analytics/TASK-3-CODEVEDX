import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- STEP 1: Load Dataset ----
df = pd.read_csv("titanic.csv")
print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ---- STEP 2: Clean Data ----
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
print("Data Cleaned!")

# ---- CHART 1: Survival Distribution ----
plt.figure(figsize=(8,5))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Distribution')
plt.xticks([0,1], ['Not Survived', 'Survived'])
plt.xlabel('Survival Status')
plt.ylabel('Count')
plt.savefig('survival_distribution.png')
plt.close()
print("Survival Distribution saved!")

# ---- CHART 2: Gender Distribution ----
plt.figure(figsize=(8,5))
df['Sex'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue','pink'])
plt.title('Gender Distribution')
plt.ylabel('')
plt.savefig('gender_distribution.png')
plt.close()
print("Gender Distribution saved!")

# ---- CHART 3: Age vs Fare Scatter ----
plt.figure(figsize=(8,5))
plt.scatter(df['Age'], df['Fare'], color='purple', alpha=0.5)
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig('age_vs_fare.png')
plt.close()
print("Age vs Fare saved!")

# ---- CHART 4: Chart Distribution ----
plt.figure(figsize=(8,5))
df['Pclass'].value_counts().plot(kind='bar', color=['blue','orange','green'])
plt.title('Passenger Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('chart_distribution.png')
plt.close()
print("Chart Distribution saved!")

# ---- CHART 5: Correlation Heatmap ----
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation Heatmap saved!")

print("\nData Visualization Complete! All 5 charts saved!")