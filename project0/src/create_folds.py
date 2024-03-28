import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('../input/mnist_train.csv')
    df["kfold"] = -1
    print(df)

    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    print(kf.split(X=df, y=df.label.values))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.label.values)):
        df.loc[val_idx, "kfold"] = fold
    
    print(df)
    df.to_csv("../input/mnist_train_folds.csv", index=False)
        
