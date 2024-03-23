# Recommendation_System_on_Yelp_Data

## Overview
The Yelp Recommendation System is designed to provide personalized business recommendations to users based on their preferences and past interactions with the platform. The system employs various collaborative filtering techniques and machine learning models to generate accurate predictions.

## Technologies Used
Programming Languages: Python and Scala
Libraries: PySpark (RDD), XGBoost
Frameworks: Apache Spark
Dataset: Subset of Yelp review dataset

## Implementation

In this project, I developed a recommendation engine using Locality Sensitive Hashing (LSH) and collaborative filtering techniques to recommend businesses to users based on their preferences. The project consists of two main tasks:

Task 1: Jaccard based LSH

Task 2: Recommendation System

### Task 1: Jaccard based LSH

#### Problem Statement:

The goal of this task is to implement Locality Sensitive Hashing (LSH) with Jaccard similarity using the Yelp dataset. The task involves identifying similar businesses based on user ratings, focusing on binary ratings (0 or 1) rather than actual star ratings. The implementation aims to efficiently find business pairs with a Jaccard similarity of at least 0.5.

#### Techniques Used:

Locality Sensitive Hashing (LSH): Implemented using Python and Apache Spark RDD to efficiently identify candidate pairs of similar businesses.

Jaccard Similarity: Calculated as the intersection divided by the union of characteristic sets.

Hash Functions: Designed a collection of hash functions to create consistent permutations of row entries in the characteristic matrix.

Signature Matrix: Constructed a signature matrix using hash functions and divided it into bands to identify candidate pairs efficiently.

#### Implementation Details:

The implementation is done in Python using PySpark for distributed computing. The main steps include:

Data Preprocessing: Loaded Yelp dataset and filtered necessary columns.

Hash Functions: Designed and implemented hash functions to create permutations of row entries.

Signature Matrix: Constructed a signature matrix and divided it into bands to identify candidate pairs efficiently.

Jaccard Similarity: Calculated Jaccard similarity for candidate pairs and filtered pairs with similarity >= 0.5.

Output: Generated a CSV file containing similar business pairs based on Jaccard similarity.

### Task 2: Recommendation System

#### Problem Statement:

The goal of this task is to build various recommendation systems using collaborative filtering techniques and the Yelp dataset. Three types of recommendation systems are implemented:

Item-based Collaborative Filtering (CF) with Pearson similarity

Model-based recommendation system using XGBoost regressor

Hybrid recommendation system combining CF and model-based approaches

#### Techniques Used:

Collaborative Filtering: Utilized user-item interactions to recommend items to users based on their behavior and preferences.

Pearson Similarity: Implemented item-based collaborative filtering using Pearson similarity to measure the correlation between item ratings.

XGBoost Regressor: Employed XGBoost regressor to train a predictive model using features extracted from the dataset.

Hybrid Approach: Combined collaborative filtering and model-based approaches to enhance recommendation accuracy.

#### Implementation Details:

Item-based Collaborative Filtering (CF):

Calculated Pearson similarity between items based on user ratings.
Predicted ratings for user-item pairs using weighted averages of similar items.
Model-based Recommendation System:

Trained an XGBoost regressor model using features extracted from the dataset.
Used the model to predict ratings for user-item pairs.

### Hybrid Recommendation System:

Combined results from CF and model-based systems using weighted averages or classification techniques.
Designed a hybrid approach to leverage the strengths of both systems and improve recommendation accuracy.

## Dataset and Evaluation
Yelp Data: Subset of Yelp review dataset with user-business interactions.

Evaluation Metrics: Precision, recall, RMSE for recommendation system performance assessment.

## Conclusion
The Yelp Recommendation System demonstrates the application of collaborative filtering and machine learning techniques for personalized business recommendations. By leveraging Spark RDD and PySpark libraries, the system efficiently processes large-scale datasets to generate accurate predictions. The hybrid approach further enhances recommendation quality, providing users with tailored suggestions based on their preferences and past interactions.

# Improvement

 The goal is to improve the performance of the recommendation system developed earlier. The primary focus is on increasing the prediction accuracy and efficiency using any method, such as hybrid recommendation systems. 

## Methodology Description:
I implemented a collaborative filtering recommendation system using a combination 
of user-based and item-based approaches. The key models employed are XGBoostRegressor 
for rating predictions and a weighted average approach for aggregation.

## Model Improvement:
1. Hyperparameter Tuning:
   I optimized the XGBoostRegressor by performing a randomized search 
   over a range of hyperparameters. This involved iterating through various 
   combinations to identify the set that minimizes RMSE and enhancing the 
   model's performance.

2. Feature Enrichment:
   To augment the model's predictive power, I incorporated additional features 
   from user.json and business.json datasets. These features include fans, friends, 
   yelping_years, and various compliment categories for users, as well as business-related 
   attributes such as total_hours, is_open, latitude, and longitude. The inclusion of 
   these features significantly improved the efficiency of the recommendation system.

Overall, the combination of hyperparameter tuning and feature enrichment contributed 
to a more accurate and efficient collaborative filtering model for predicting user ratings. 

The incorporation of hybrid recommendation systems and feature enrichment further enhances recommendation quality, offering users tailored suggestions based on their preferences and past interactions.

Error distribution:
>=0 and <1: 101857
>=1 and <2: 33300
>=2 and <3: 6116
>=3 and <4: 771
>=4: 0

RMSE: 0.9780393799890853

Execution time: 1342.943686246872 seconds
