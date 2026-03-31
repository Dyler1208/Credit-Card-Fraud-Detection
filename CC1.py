import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv("credit_card_frauds.csv")




df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour


df["dob"] = pd.to_datetime(df["dob"])
df["age"] = pd.Timestamp.now().year - df["dob"].dt.year


df = df[['amt', 'category', 'city_pop', 'age', 'hour', 'is_fraud']]


le_category = LabelEncoder()
df["category"] = le_category.fit_transform(df["category"])


X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_category, open("le_category.pkl", "wb"))


pickle.dump(list(le_category.classes_), open("categories.pkl", "wb"))

print("✅ Model Ready!")
