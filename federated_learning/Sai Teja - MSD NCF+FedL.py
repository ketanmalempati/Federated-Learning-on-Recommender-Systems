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

taste_profile_df = pd.read_csv('C:\\Users\\Teja\Desktop\\train_triplets.txt', sep='\t', header=None, names = ['user_id','song_id','play_count'], nrows = 2000000)
print(taste_profile_df.shape)
print(taste_profile_df.head(8))
print("\n\nSummary of column statistics: \n", taste_profile_df.describe())

taste_profile_df = taste_profile_df.drop(taste_profile_df[taste_profile_df.play_count > 5].index)
print(taste_profile_df.shape)

taste_profile_df = taste_profile_df.dropna()
taste_profile_df.drop_duplicates(inplace=True)

taste_profile = taste_profile_df.groupby('play_count', group_keys=False).apply(lambda x: x.sample(int((len(x)/len(taste_profile_df))*150000)))
tasteprofile_data = taste_profile[["user_id", "song_id", "play_count"]]
print(tasteprofile_data.shape)

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


tasteprofile_data["user"] = tasteprofile_data["user_id"]
tasteprofile_data["label"] = tasteprofile_data["play_count"]
#, "item", "label", "time"
tasteprofile_data["item"] = tasteprofile_data["song_id"]
tasteprofile_data=tasteprofile_data[["user","item","label"]]

train_data, eval_data, test_data = random_split(tasteprofile_data, multi_ratios=[0.8, 0.1, 0.1])


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
ncf_m.save(r"C:\\Users\\Teja\Desktop\\","ncf")

datanp = np.load(r"C:\\Users\\Teja\Desktop\\ncf_tf_variables.npz")

n = 5  
split_data = np.array_split(tasteprofile_data, n)

from libreco.data import split_by_ratio, DataInfo, DatasetPure
from libreco.algorithms import NCF
from libreco.evaluation import evaluate
import os

# Create a directory for the models if it doesn't already exist
model_dir = r"C:\\Users\\Teja\Desktop\\ncfmodels_msd"
os.makedirs(model_dir, exist_ok=True)
# Initialize storage for evaluation metrics
evaluation_results = []
data_infos= []
train_data, main_data_info = DatasetPure.build_trainset(tasteprofile_data)
import tensorflow as tf
graph = tf.Graph()

 
for i, part in enumerate(split_data):
    # Prepare the dataset for training
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    train_data, eval_data, test_data = random_split(part, multi_ratios=[0.8, 0.1, 0.1])
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)
    data_info = main_data_info
    data_infos.append(data_info)
    keras.backend.clear_session()

    # Initialize and train the NCF model
    ncf = NCF(
        task='ranking',
        data_info=data_info,
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler='random',
        num_neg=1,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None
    )

    ncf.fit(
        train_data,
        neg_sampling=True,
        eval_data=eval_data,
        
        metrics=["loss"],
        verbose=1,
        shuffle=False,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        num_workers=0
    )


    # Save the model
    ncf_loaded=ncf
    ncf.save(model_dir, f"ncf_model_part_{i}")
for idx, result in enumerate(evaluation_results):
    print(f"Results for model {idx}:")
    for metric, value in result.items():
        print(f"{metric}: {value}")
    
num_models = 5
# Lists to store paths
recs_files = [os.path.join(model_dir, f"ncf_model_part_{i}_default_recs.npz") for i in range(num_models)]
vars_files = [os.path.join(model_dir, f"ncf_model_part_{i}_tf_variables.npz") for i in range(num_models)]        
def aggregate_npz_files_adjusted(file_list):
    # Find minimum sizes for each key across all files
    min_sizes = {}
    for file_path in file_list:
        with np.load(file_path) as data:
            for key in data.files:
                array_shape = data[key].shape
                if key not in min_sizes:
                    min_sizes[key] = array_shape
                else:
                    # Compare each dimension and keep the minimum
                    existing_shape = min_sizes[key]
                    min_sizes[key] = tuple(min(existing_shape[dim], array_shape[dim]) for dim in range(len(array_shape)))

    # Initialize the data structure for summing arrays
    aggregated_data = {}
    for key, size in min_sizes.items():
        if len(size) == 1:
            aggregated_data[key] = np.zeros(size[0])  # Handle 1D arrays
        elif len(size) > 1:
            aggregated_data[key] = np.zeros(size)  # Handle 2D and potentially higher dimensions arrays
        else:
            aggregated_data[key] = 0

    # Sum and average arrays, trimming arrays to min size
    count = len(file_list)
    for file_path in file_list:
        with np.load(file_path) as data:
            for key in data.files:
                # Create a slice object to trim the array appropriately
                #array_slice = tuple(slice(0, min_dim) for min_dim in min_sizes[key])
                trimmed_array = data[key]
                aggregated_data[key] += trimmed_array / count

    return aggregated_data

import time

# Start time
start_time = time.time()
aggregated_recs = aggregate_npz_files_adjusted(recs_files)
aggregated_vars = aggregate_npz_files_adjusted(vars_files)
end_time = time.time()

# Calculate duration
duration = end_time - start_time
print(f"The method took {duration:.2f} seconds to run.")

# Saving the aggregated data into new npz files
np.savez(os.path.join(model_dir, 'aggregated_default_recs.npz'), **aggregated_recs)
np.savez(os.path.join(model_dir, 'aggregated_tf_variables.npz'), **aggregated_vars)
model_path = os.path.join(model_dir, "aggregated")
k = ncf.output

tf.compat.v1.reset_default_graph()
# Create a new instance of NCF and load the saved model
nfc_m = NCF(
sampler = "unconsumed",
seed = 0,
task="rating",
data_info=data_info,
embed_size=16,
n_epochs=10,
lr=1e-3,
batch_size=2048,
num_neg=1
)  # Adjust parameters as needed

nfc_m.load(model_name = "aggregated",data_info=data_info,path = model_dir)
nfc_m.output = k
print("Done")

from libreco.evaluation import evaluate

preds = []
actual = []
for i, row in tasteprofile_data.iterrows():
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



from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json  # Extract the JSON data sent
    return jsonify({"received": True, "data": str(aggregated_vars)}), 200

if __name__ == '__main__':
    app.run(debug=False, port=5000)




