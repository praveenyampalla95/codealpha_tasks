import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset.csv")

print(df.head())
print(df.info())
# Bar Chart - Survival Count

df["Survived"].value_counts().plot(kind="bar")

plt.title("Survival Count")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.ylabel("Number of Passengers")
plt.savefig("survival_count.png")


plt.show()
# Pie Chart - Gender Distribution

df["Sex"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Gender Distribution")
plt.savefig("gender_distribution.png")


plt.show()
# Histogram - Age Distribution

plt.hist(df["Age"].dropna())

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("age_distribution.png")


plt.show()
# Count Plot - Passenger Class Distribution

sns.countplot(x="Pclass", data=df)

plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.savefig("pclass_distribution.png")

plt.show()
# Heatmap - Correlation between numerical columns

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")


plt.show()