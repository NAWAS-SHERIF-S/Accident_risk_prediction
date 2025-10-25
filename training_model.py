

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt


print("Loading preprocessed datasets...")
X_train = pd.read_csv(r"D:\AIML Hackathon\X_train_processed.csv")
X_test = pd.read_csv(r"D:\AIML Hackathon\X_test_processed.csv")
y_train = pd.read_csv(r"D:\AIML Hackathon\y_train_processed.csv").values.ravel()
y_test = pd.read_csv(r"D:\AIML Hackathon\y_test_processed.csv").values.ravel()

print(f"✅ Data loaded: {X_train.shape[0]} training rows, {X_test.shape[0]} testing rows")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,          # number of boosting rounds (trees)
    learning_rate=0.05,        # step size shrinkage
    max_depth=6,               # tree depth
    subsample=0.8,             # fraction of data used per tree
    colsample_bytree=0.8,      # fraction of features per tree
    eval_metric='mlogloss',    # multi-class classification metric
    random_state=42
)

print("\n Training XGBoost model...")
xgb_model.fit(X_train, y_train)

print(" Model training completed!")

print("\n Evaluating model...")
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

xgb.plot_importance(xgb_model)
plt.title("Feature Importance - SafeRoads XGBoost")
plt.show()

joblib.dump(xgb_model, r"D:\AIML Hackathon\saferoads_xgb_model.pkl")
print("\nModel saved as saferoads_xgb_model.pkl in D:\\AIML Hackathon")

