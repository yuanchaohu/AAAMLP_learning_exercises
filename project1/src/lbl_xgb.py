import pandas as pd 
import xgboost as xgb

from sklearn import metrics, preprocessing

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimator=200
    )
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold in range(5):
        run(fold)