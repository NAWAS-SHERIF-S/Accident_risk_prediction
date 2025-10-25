import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


df = pd.read_csv(r"D:\AIML Hackathon\saferoads_accident_risk_dataset.csv")

print("✅ Dataset loaded successfully!")
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


df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')


categorical_cols = [
    'weather', 'road_type', 'road_condition',
    'time_of_day', 'day_of_week', 'light_condition', 'risk_level'
]

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

numeric_cols = [
    col for col in df.select_dtypes(include=['float64', 'int64']).columns
    if col not in ['risk_level']
]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop(['risk_level', 'timestamp'], axis=1)
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

X_train.to_csv(r"D:\AIML Hackathon\X_train_processed.csv", index=False)
X_test.to_csv(r"D:\AIML Hackathon\X_test_processed.csv", index=False)
y_train.to_csv(r"D:\AIML Hackathon\y_train_processed.csv", index=False)
y_test.to_csv(r"D:\AIML Hackathon\y_test_processed.csv", index=False)

print("\nData preprocessing complete.")
print("Files saved in: D:\\AIML Hackathon")
