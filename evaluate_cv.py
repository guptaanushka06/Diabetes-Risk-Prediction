from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessing
from src.config import TARGET_COLUMN


def main():
    loader = DataLoader()
    df = loader.load_dataset()
    df = loader.remove_duplicates()
    df = DataPreprocessing().engineer_features(df)

    X, y = df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]
    # use a 30,000-row sample so it finishes in minutes
    X, _, y, _ = train_test_split(X, y, train_size=30000, stratify=y, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    scoring = ["accuracy", "f1_macro", "recall_macro", "precision_macro", "roc_auc_ovr_weighted"]

    for name, m in models.items():
        print(f"Running {name}...", flush=True)
        pipe = Pipeline([
            ("smote", BorderlineSMOTE(random_state=42)),
            ("scale", StandardScaler()),
            ("model", m),
        ])
        r = cross_validate(pipe, X, y, cv=cv, n_jobs=-1, scoring=scoring)
        print(f"{name}: " + " | ".join(
            f"{k[5:]} {v.mean():.3f}±{v.std():.3f}" for k, v in r.items() if k.startswith("test_")
        ), flush=True)


if __name__ == "__main__":
    main()