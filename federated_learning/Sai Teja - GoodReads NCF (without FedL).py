import pandas as pd
import json
import gzip
import numpy as np
import random
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SVD,SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering

from surprise.model_selection import GridSearchCV
from tqdm import tqdm
import matplotlib.pyplot as plt
from surprise.model_selection.validation import cross_validate

print("Imported all libraries")

books_df = pd.read_csv("C:\\Users\\Teja\Desktop\\gri.csv")
print("Data loaded")

print(books_df.shape)
print(books_df.head())

books_df = books_df.drop(books_df[books_df['is_read'] == 0].index)
books_df = books_df.drop(books_df[books_df['is_reviewed'] == 0].index)
books = books_df.groupby('rating', group_keys=False).apply(lambda x: x.sample(int((len(x)/len(books_df))*75000)))

print(books.shape)

booksdata = books[['user_id', 'book_id', 'rating']]
print(booksdata.head())

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input, Embedding, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras
from keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten
from keras import regularizers
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from libreco.data import random_split, DatasetPure
from libreco.algorithms import NCF  # pure data, 
from libreco.evaluation import evaluate


booksdata["user"] = booksdata["user_id"]
booksdata["label"] = booksdata["rating"]
#, "item", "label", "time"
booksdata["item"] = booksdata["book_id"]
booksdata=booksdata[["user","item","label"]]

train_data, eval_data, test_data = random_split(booksdata, multi_ratios=[0.8, 0.1, 0.1])


train_data, data_info= DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)

ncf_m = NCF(
    task="rating",
    data_info=data_info,
    loss_type="cross_entropy",
    embed_size=16,
    n_epochs=10,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
)

ncf_m.fit(
    train_data,
    neg_sampling=False, #for rating, this param is false else True
    verbose=2,
    eval_data=eval_data,
    metrics=["loss"],
)

# do final evaluation on test data
evaluate(
    model=ncf_m,
    data=test_data,
    neg_sampling=False,
    metrics=["loss"],
)

evaluate(
    model=ncf_m,
    data=test_data,
    neg_sampling=False,
    metrics=["mae"],
)

evaluate(
    model=ncf_m,
    data=test_data,
    neg_sampling=False,
    metrics=["rmse"],
)


from libreco.evaluation import evaluate

preds = []
actual = []
for i, row in booksdata.iterrows():
    preds.append(round(ncf_m.predict(row["user"],row['item'])[0]))
    actual.append(round(row['label']))

import numpy as np
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support
rmse = np.sqrt(mean_squared_error(actual, preds))
print(f"RMSE: {rmse}")


precision, recall, f1_score, _ = precision_recall_fscore_support(actual, preds, average='macro')
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")




