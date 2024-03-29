import pandas as pd 

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    df = df.drop(num_cols, axis=1)
    print(f"df original shape: {df.shape}")

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data)

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    print(f"train dataset shape: {x_train.shape}")

    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    print(f"fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold in range(5):
        run(fold)