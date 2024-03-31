import pandas as pd 

from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    def evalute_score(self, X, y):
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    
    def _feature_selection(self, X, y):
        good_features = []
        best_scores = []

        num_features = X.shape[1]

        while True:
            this_feature = None 
            best_score = 0

            for feature in range(num_features):
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                xtrain = X[:, selected_features]
                score = self.evalute_score(xtrain, y)
                if score > best_score:
                    this_feature = feature
                    best_score = score
                
                if this_feature != None:
                    good_features.append(this_feature)
                    best_scores.append(best_score)
                
                if len(best_scores) > 2:
                    if best_scores[-1] < best_scores[-2]:
                        break
        
        return best_scores[:-1], good_features[:-1]
    
    def __call__(self, X, y):
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores

if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=100)
    X_transformed, scores = GreedyFeatureSelection()(X, y)