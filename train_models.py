import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# --- Load Dataset ---
df = pd.read_csv("heart.csv")

# --- Encode categorical data (if any) ---
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# --- Split data ---
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision Tree ---
print("\nTraining Decision Tree...")
dtree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
}
grid = GridSearchCV(dtree, param_grid, cv=5)
grid.fit(x_train, y_train)
best_tree = grid.best_estimator_
print("Decision Tree Accuracy:", accuracy_score(y_test, best_tree.predict(x_test)))
pickle.dump(best_tree, open("tree.pkl", "wb"))

# --- Logistic Regression ---
print("\nTraining Logistic Regression...")
logreg = LogisticRegression(max_iter=1000)
logreg.fit(x_train, y_train)
pickle.dump(logreg, open("logisticRegression.pkl", "wb"))

# --- Random Forest ---
print("\nTraining Random Forest...")
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
pickle.dump(rf, open("RandomForest.pkl", "wb"))

# --- SVM ---
print("\nTraining SVM...")
best_kernel = None
best_score = 0
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    svm = SVC(kernel=k)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    score = f1_score(y_test, y_pred, average="weighted")
    if score > best_score:
        best_score = score
        best_kernel = k
svm = SVC(kernel=best_kernel, probability=True)
svm.fit(x_train, y_train)
pickle.dump(svm, open("SVM.pkl", "wb"))

print("\n✅ All models trained and saved successfully!")
