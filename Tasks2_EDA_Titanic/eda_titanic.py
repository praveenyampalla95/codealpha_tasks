import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("train.csv")

# Show first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Summary statistics
print(df.describe())

# Missing values
print("\nMissing Values")
print(df.isnull().sum())

# Survival Count
print("\nSurvival Count")
print(df["Survived"].value_counts())

# Survival by Gender
print("\nSurvival by Gender")
print(pd.crosstab(df["Sex"], df["Survived"]))

# Survival Count Chart
sns.countplot(x="Survived", data=df)

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.savefig("survival_count.png")

plt.show()


# Survival by Gender Chart
sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.savefig("gender_survival.png")

plt.show()