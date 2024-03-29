import os, gc, joblib
import pandas as pd 
import numpy as np 
from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K 

def create_model(data, catcols):
    inputs = []
    outputs = []
    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_values/2), 50))

        inp = layers.Input(shape=(1,))
        out = layers.Embedding(
            num_unique_values+1, embed_dim, name=c
        )(inp)
        out = layers.Reshape(target_shape=(embed_dim,))(out)
        inputs.append(inp)
        outputs.append(out)
    
    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=y)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model

def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)

    features = [
        c for c in df.columns if c not in num_cols and c not in ("kfold", "income")
    ]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    for col in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, col] = lbl_enc.fit_transform(df[col].values)
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = create_model(df, features)

    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]
    ytrain = df_train.income.values 
    yvalid = df_valid.income.values

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(
        xtrain,
        ytrain_cat,
        validation_data = (xvalid, yvalid_cat),
        verbose=1,
        batch_size=1024,
        epochs=3
    )

    valid_preds = model.predict(xvalid)[:, 1]
    print(metrics.roc_auc_score(yvalid, valid_preds))
    K.clear_session()

if __name__ == "__main__":
    for fold in range(5):
        run(fold)