# CMPE295B_Federated-Learning-on-Recommender-Systems
## Introduction
![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/6b7a6a12-df9a-4960-aaa2-210727646e9d)
<p align="justify"> 
Recommender Systems have greatly affected how we consume services, products, and content
in recent years. In the digital world, they are essential for users as well as companies and are
widely used to influence user choices in products, food, music, movies, content, and many other
things. They have various applications in everyday life, such as clothing, restaurants, or song
recommendations, recommending friends and connections on social media platforms, and
rating predictions for e-commerce and businesses to help them understand user choices well
and ultimately increase user retention. Existing techniques for recommendation systems include
collaborative filtering, context-based recommendations, matrix factorization, and many more.
While these systems have enhanced user experience, customer satisfaction, and business
models, there are concerns about data privacy. Typically, these systems collect private
information about users based on their online behavior, cookies, and social interactions, such as
user clicks, time, and other data points to improve recommendations. This centralized approach
to collecting and storing information is prone to privacy risks and data breaches. In this project,
we propose federated learning for recommender systems to handle data privacy issues across
seven distinct datasets to benchmark the performance of federated learning and provide
insights into the efficacy of the approach in preserving privacy while maintaining recommender
system performance. In this approach, the models are trained on edge devices using the data
on user machines. This technique shares each user’s updated parameters using optimal
aggregation functions instead of actual data to a shared server. This decentralized way ensures
that the data remains local and protects data privacy. Federated learning setups showed no
change when ranking recommendations but showed a slight decrease in efficiency when it
comes to predicting the ratings compared to traditional centralized approaches, which is
expected as a significantly smaller dataset is used, in-turn we achieved complete data privacy
and security. Overall, this approach is useful to achieve better user trust, experiences, and
recommendations. </p>

## Datasets
![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/6379600e-e51e-4b46-a8eb-9ac9b891c7d8)

### Clothes
https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit
### Food
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
### MarketBias
https://cseweb.ucsd.edu/~jmcauley/datasets.html#market_bias
### Amazon
https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
### Google Review
https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local
### GoodReads
https://mengtingwan.github.io/data/goodreads.html
### Spotify
http://millionsongdataset.com/

## Architecture Diagram
![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/3518f2d7-3c1b-404d-bd89-f5832d904998)

## Research Questions
- RQ1: How do model results differ with and without the implementation of federated learning?
- RQ2: Are there observable differences in results based on different types of datasets?
- RQ3: Does the choice of aggregation function affect the outcomes significantly?
- RQ4: Which models perform the best across all datasets or specific types of data?
- RQ5: Does the federated fit have an affect on the results?


## Results
Comparing different Federated Averaging methods:

- SVD:
  - F1 Change (Avg): -2.61%
  - F1 Change (Weighted): -2.53%
  - RMSE Change (Avg): 25.35%
  - RMSE Change (Weighted): 25.86%
- SVDpp:
  - F1 Change (Avg): -3.18%
  - F1 Change (Weighted): -3.13%
  - RMSE Change (Avg): 24.13%
  - RMSE Change (Weighted): 25.09%
- NMF:
  - F1 Change (Avg): -3.31%
  - F1 Change (Weighted): -3.57%
  - RMSE Change (Avg): 50.51%
  - RMSE Change (Weighted): 51.42%
- NCF:
  - F1 Change (Avg): -33.05%
  - F1 Change (Weighted): -28.84%
  - RMSE Change (Avg): 24.07%
  - RMSE Change (Weighted): 25.15%

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/d6576e25-eaa3-4512-9062-1449b9e05948)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/5f35198e-c6e3-4846-9498-d1b09321672a)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/d2a17f19-0a8e-4f98-82ac-e1c0f1b4a815)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/ae7ac7f3-bc25-40a6-beb3-73ff2b4e7884)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/fa4a15ed-455c-4306-b223-78af4e0e295e)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/0555980a-d40d-4ea1-a930-00ac78fc62b4)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/2309d5c9-6f2e-4a99-9c8a-6f3e469ffaab)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/495b9f75-17c5-4aec-bd36-062701ab4dac)

![image](https://github.com/ketanmalempati/CMPE295B_Federated-Learning-on-Recommender-Systems/assets/57043103/44a78d18-0c97-411a-8cbf-d768c6aa5959)




