from re import A
import numpy as np 

from functools import partial
from scipy.optimize import fmin
from sklearn import metrics

class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0
    
    def _auc(self, coef, X, y):
        x_coef = X[:, np.newaxis] * coef

        predictions = np.sum(x_coef, axis=1)

        auc_score = metrics.roc_auc_score(y, predictions)

        return -1 * auc_score
    
    def fit(self, X, y):
        loss_partial = partial(self._auc, X, y)

        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)
    
    def predict(self, X):
        x_coef = X[:, np.newaxis] * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

import xgboost as xgb 
from sklearn.datasets import make_classification
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

X, y = make_classification(n_samples=10000, n_features=25)
xfold1, xfold2, yfold1, yfold2 = model_selection.train_test_split(
    X, 
    y,
    test_size=0.5,
    stratify=y
)

def ensemble_models(xtrain, ytrain, xtest, ytest):
    # ensemble models
    logreg = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier()
    xgbc = xgb.XGBClassifier()

    logreg.fit(xtrain, ytrain)
    rf.fit(xtrain, ytrain)
    xgbc.fit(xtrain, ytrain)

    pred_logreg = logreg.predict_proba(xtest)[:, 1]
    pred_rf = rf.predict_proba(xtest)[:, 1]
    pred_xgbc = xgbc.predict_proba(xtest)[:, 1]

    # average
    avg_pred = (pred_logreg + pred_rf + pred_xgbc) / 3

    fold2_preds = np.column_stack((
        pred_logreg,
        pred_rf,
        pred_xgbc,
        avg_pred
    ))

    aucs_fold2 = []
    for i in range(fold2_preds.shape[1]):
        auc = metrics.roc_auc_score(ytest, fold2_preds[:, i])
        aucs_fold2.append(auc)

    print(f"fold-test: LR AUC = {aucs_fold2[0]}")
    print(f"fold-test: RF AUC = {aucs_fold2[1]}")
    print(f"fold-test: XGB AUC = {aucs_fold2[2]}")
    print(f"fold-test: average pred auc = {aucs_fold2[3]}")

    return aucs_fold2, fold2_preds

aucs_fold2, fold2_preds = ensemble_models(xfold1, yfold1, xfold2, yfold2)
aucs_fold1, fold1_preds = ensemble_models(xfold2, yfold2, xfold1, yfold1)

opt = OptimizeAUC()
opt.fit(fold1_preds[:, :-1], yfold1)
opt_preds_fold2 = opt.predict(fold2_preds[:, :-1])
auc = metrics.roc_auc_score(yfold2, opt_preds_fold2)
print(f"optimized AUC: fold 2 = {auc}")
print(f"coefficients = {opt.coef_}")

opt = OptimizeAUC()
opt.fit(fold2_preds[:, :-1], yfold2)
opt_preds_fold1 = opt.predict(fold1_preds[:, :-1])
auc = metrics.roc_auc_score(yfold1, opt_preds_fold1)
print(f"optimized auc, fold 1 = {auc}")
print(f"coefficients = {opt.coef_}")