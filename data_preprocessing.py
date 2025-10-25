import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import pickle


df = pd.read_csv("saferoads_accident_risk_dataset.csv")
print("Dataset loaded successfully!")
print("Initial shape:", df.shape)
print(df.head())



df.fillna({
    'weather': df['weather'].mode()[0],
    'road_type': df['road_type'].mode()[0],
    'road_condition': df['road_condition'].mode()[0],
    'time_of_day': df['time_of_day'].mode()[0],
    'light_condition': df['light_condition'].mode()[0],
    'visibility_km': df['visibility_km'].mean(),
    'traffic_density': df['traffic_density'].mean()
}, inplace=True)


if 'timestamp' in df.columns:
    original_count = len(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    invalid_count = df['timestamp'].isna().sum()
    


categorical_cols = [
    'weather', 'road_type', 'road_condition',
    'time_of_day', 'day_of_week', 'light_condition', 'risk_level'
]

encoders = {}
for col in categorical_cols:
    if col in df.columns and col != 'risk_level':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Handle risk_level separately
if 'risk_level' in df.columns:
    risk_encoder = LabelEncoder()
    df['risk_level'] = risk_encoder.fit_transform(df['risk_level'])
    encoders['risk_level'] = risk_encoder

numeric_cols = [
    col for col in df.select_dtypes(include=['float64', 'int64']).columns
    if col not in ['risk_level']
]

scaler = StandardScaler()
df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Drop timestamp if it exists, otherwise just risk_level
columns_to_drop = ['risk_level']
if 'timestamp' in df.columns:
    columns_to_drop.append('timestamp')
X = df.drop(columns_to_drop, axis=1)
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter preprocessing:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Target distribution:\n", y_train.value_counts())

output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, "X_train_processed.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test_processed.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train_processed.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test_processed.csv"), index=False)

# Save encoders and scaler
with open(os.path.join(output_dir, "encoders.pkl"), 'wb') as f:
    pickle.dump(encoders, f)
with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

print("\nData preprocessing complete.")
print(f"Files saved in: {os.path.abspath(output_dir)}")
print("Encoders and scaler saved for model inference.")
print(f"Feature columns: {list(X.columns)}")
