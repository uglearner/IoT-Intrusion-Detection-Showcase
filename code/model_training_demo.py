import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load tiny sample dataset
df = pd.read_csv("../data/sample_dataset.csv")

# Split features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train simple model
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)

print("Accuracy on sample dataset:", clf.score(X_test, y_test))
