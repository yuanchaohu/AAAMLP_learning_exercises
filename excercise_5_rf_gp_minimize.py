import numpy as np 
import pandas as pd 

from functools import partial
from sklearn import ensemble, metrics, model_selection
from skopt import gp_minimize, space

def optimize(params, param_names, x, y):
    pamrams = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

        fold_accuracy = metrics.accuracy_score(
            ytest,
            preds
        )
        accuracies.append(fold_accuracy)
    return -1 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("./datasets/mobile_price/train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]

    opt_func = partial(
        optimize,
        param_names=param_names,
        x=X,
        y=y
    )

    result = gp_minimize(
        opt_func,
        dimensions = param_space,
        n_calls = 15,
        n_random_starts=10,
        verbose=10
    )

    best_params = dict(
        zip(param_names, result.x)
    )

    print(best_params)