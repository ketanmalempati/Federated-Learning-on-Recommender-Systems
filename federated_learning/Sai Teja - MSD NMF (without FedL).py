import pandas as pd
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
print(taste_profile_df.shape)


import pandas as pd

# Example DataFrame loading
# df1 = pd.read_csv('path_to_dataset_1.csv')
# df2 = pd.read_csv('path_to_dataset_2.csv')

# Function to calculate the metrics
def calculate_metrics(df):
    num_users = df['user_id'].nunique()
    num_items = df['song_id'].nunique()
    num_ratings = len(df)
    total_possible_ratings = num_users * num_items
    sparsity = 1 - (num_ratings / total_possible_ratings)
    avg_rating = df['play_count'].mean()
    std_rating = df['play_count'].std()
    avg_ratings_per_user = num_ratings / num_users
    avg_ratings_per_item = num_ratings / num_items
    
    return {
        'Number of Users': num_users,
        'Number of Items': num_items,
        'Number of Ratings': num_ratings,
        'Sparsity': sparsity,
        'Average Rating': avg_rating,
        'Rating STD': std_rating,
        'Avg Ratings per User': avg_ratings_per_user,
        'Avg Ratings per Item': avg_ratings_per_item
    }

# Calculate metrics for each dataset
metrics_df1 = calculate_metrics(taste_profile_df)

# Creating a DataFrame to hold the summary table
summary_df = pd.DataFrame([metrics_df1])

# Display the summary table
print(summary_df)

taste_profile = taste_profile_df.groupby('play_count', group_keys=False).apply(lambda x: x.sample(int((len(x)/len(taste_profile_df))*150000)))
tasteprofile_data = taste_profile[["user_id", "song_id", "play_count"]]

print("\n\nSummary of column statistics: \n", tasteprofile_data.describe())
print("continuing....")

reader = Reader(rating_scale=(1, 5))

from surprise.model_selection import train_test_split

nmf_params = {
        "n_factors": [10],
    "n_epochs": [10]
    }

data = Dataset.load_from_df(taste_profile[["user_id", "song_id", "play_count"]], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

algo = NMF(n_factors=15,
n_epochs=50,
biased=False,
reg_pu=0.06,
reg_qi=0.06,
init_low=0,  
init_high=1,
random_state=42,  
verbose=True)

# svd_params = {
#         "n_factors": [10],
#     "n_epochs": [10]
#     }

# data = Dataset.load_from_df(taste_profile[["user_id", "song_id", "play_count"]], reader)
# trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# algo = SVD(n_factors=15,
# n_epochs=50,
# reg_pu=0.06,
# reg_qi=0.06,
# random_state=42,  # Fixed random state for reproducibility
# verbose=True)

# svdpp_params = {
#     "n_factors": [10],
#     "n_epochs": [10]
#     }

# data = Dataset.load_from_df(taste_profile[["user_id", "song_id", "play_count"]], reader)
# trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# algo = SVDpp(n_factors=15,
# n_epochs=50,
# reg_pu=0.06,
# reg_qi=0.06,
# random_state=42,  # Fixed random state for reproducibility
# verbose=True)


algo.fit(trainset)
predictions = algo.test(testset)
print(accuracy.rmse(predictions))

from collections import defaultdict

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls
def f1_score(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict



# Compute Precision and Recall at k=10
precisions, recalls = precision_recall_at_k(predictions, k=1, threshold=1)

# Average precision and recall over all users
avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

# Compute F1-score
f1 = f1_score(avg_precision, avg_recall)

print(f'Precision: {avg_precision}')
print(f'Recall: {avg_recall}')
print(f'F1 Score: {f1}')


predictions = [algo.predict(uid, iid, r_ui=rating) for (uid, iid, rating) in testset]
# Organize predictions by user
user_ratings = defaultdict(list)
for prediction in predictions:
    uid = prediction.uid
    iid = prediction.iid
    est = prediction.est
    r_ui = prediction.r_ui
    user_ratings[uid].append((iid, r_ui, est))

# Sort by actual and predicted ratings to obtain rankings
user_actual_rankings = {}
user_predicted_rankings = {}
for uid, ratings in user_ratings.items():
    user_actual_rankings[uid] = sorted(ratings, key=lambda x: x[1], reverse=True)
    user_predicted_rankings[uid] = sorted(ratings, key=lambda x: x[2], reverse=True)

# Determine item relevance scores
relevancy = {}
from sklearn.metrics import ndcg_score
ndcg_scores=[]
ndcg_penality = -3
for uid in user_actual_rankings:
    actual_rank = {item: idx for idx, (item, _, _) in enumerate(user_actual_rankings[uid])}
    predicted_rank = {item: idx for idx, (item, _, _) in enumerate(user_predicted_rankings[uid])}
    try:
        ndcg_scores.append(ndcg_score(actual_rank.values(),predicted_rank.values()))
    except:
        if list(actual_rank.values())[0]==list(predicted_rank.values())[0]:
            ndcg_scores.append(1)
        else:
            ndcg_scores.append(ndcg_penality)
# Prepare for nDCG calculation
average_ndcg = np.mean(ndcg_scores)
print(f"Average nDCG: {average_ndcg}")