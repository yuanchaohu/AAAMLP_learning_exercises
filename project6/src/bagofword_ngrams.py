import pandas as pd 

from nltk.tokenize import word_tokenize
from sklearn import linear_model, naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if __name__ == "__main__":
    df = pd.read_csv("./datasets/imdb/Dataset.csv")
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x=="positive" else 0
    )

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.sentiment.values

    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f
    
    for fold in range(5):
        train_df = df[df.kfold != fold].reset_index(drop=True)
        test_df = df[df.kfold == fold].reset_index(drop=True)

        count_vec = TfidfVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None,
            ngram_range=(1, 3)
        )
        count_vec.fit(train_df.review)

        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        # model = linear_model.LogisticRegression()
        model = naive_bayes.MultinomialNB()
        model.fit(xtrain, train_df.sentiment)

        preds = model.predict(xtest)
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold}; Accuracy: {accuracy}")
