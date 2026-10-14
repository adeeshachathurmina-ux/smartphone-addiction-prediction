import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_and_save_model():
    print("Loading dataset...")
    df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")

    # Clean data similar to notebook
    cols_to_drop = [c for c in ['transaction_id', 'user_id', 'addiction_level'] if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    X = df_encoded.drop('addicted_label', axis=1)
    y = df_encoded['addicted_label']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Models
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    # Evaluate
    log_pred = log_model.predict(X_test_scaled)
    knn_pred = knn_model.predict(X_test_scaled)

    log_f1 = f1_score(y_test, log_pred)
    knn_f1 = f1_score(y_test, knn_pred)

    print(f"Logistic Regression F1: {log_f1:.4f}, Accuracy: {accuracy_score(y_test, log_pred):.4f}")
    print(f"KNN F1: {knn_f1:.4f}, Accuracy: {accuracy_score(y_test, knn_pred):.4f}")

    if log_f1 >= knn_f1:
        best_model = log_model
        best_name = "Logistic Regression"
    else:
        best_model = knn_model
        best_name = "KNN"

    print(f"Selected Best Model: {best_name}")

    # Save artifacts
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "model_columns.pkl")
    print("Artifacts saved: best_model.pkl, scaler.pkl, model_columns.pkl")

if __name__ == "__main__":
    train_and_save_model()
