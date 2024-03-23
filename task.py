'''
Methodology Description:
I implemented a collaborative filtering recommendation system using a combination 
of user-based and item-based approaches. The key models employed are XGBoostRegressor 
for rating predictions and a weighted average approach for aggregation.

Model Improvement:
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

Error distribution:
>=0 and <1: 101857
>=1 and <2: 33300
>=2 and <3: 6116
>=3 and <4: 771
>=4: 0

RMSE: 0.9780393799890853

Execution time: 1342.943686246872 seconds

'''
from pyspark import SparkContext, SparkConf
import json, time, sys
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


def init_spark_context(app_name):
    return SparkContext(appName=app_name).getOrCreate()

def calcs(b, u, u_b_dict, b_u_dict, b_avg_dict, u_avg_dict, bur_dict, w_dict):
    if u not in u_b_dict:
        return 3.5
    if b not in b_u_dict:
        return u_avg_dict[u]
    w_list = []
    for b1 in u_b_dict[u]:
        temp = tuple(sorted((b1, b)))
        w = w_dict.get(temp)
        if w is None:
            w = calculate_weight(b, b1, u, bur_dict, b_avg_dict, b_u_dict)
            w_dict[temp] = w
        w_list.append((w, float(bur_dict[b1][u])))
    return calcr(w_list)

def calculate_weight(b, b1, u, bur_dict, b_avg_dict, b_u_dict):
    u_inter = b_u_dict[b] & b_u_dict[b1]
    if len(u_inter) <= 1:
        return (5.0 - abs(b_avg_dict[b] - b_avg_dict[b1])) / 5
    return calculate_weight_with_interactions(b, b1, u, u_inter, bur_dict)

def calculate_weight_with_interactions(b, b1, u, u_inter, bur_dict):
    u_inter = list(u_inter)
    weights = [(5.0 - abs(float(bur_dict[b][u_inter[i]]) - float(bur_dict[b1][u_inter[i]])) / 5) for i in range(2)]
    return sum(weights) / 2



def calcr(w_list):
    w_list_can = sorted(w_list, key=lambda x: -x[0])[:15]
    X = sum(w * r for w, r in w_list_can)
    Y = sum(abs(w) for w, _ in w_list_can)
    return X / Y if Y != 0 else 3.5

def process_traindata(train_text):
    b_u_train = train_text.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).repartition(100)
    b_u_dict = dict(b_u_train.collect())

    u_b_train = train_text.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).repartition(100)
    u_b_dict = dict(u_b_train.collect())

    b_avg = train_text.map(lambda row: (row[0], float(row[2]))).groupByKey().mapValues(list).map(
        lambda x: (x[0], sum(x[1]) / len(x[1]))).repartition(100)
    b_avg_dict = dict(b_avg.collect())

    u_avg = train_text.map(lambda row: (row[1], float(row[2]))).groupByKey().mapValues(list).map(
        lambda x: (x[0], sum(x[1]) / len(x[1]))).repartition(100)
    u_avg_dict = dict(u_avg.collect())

    b_u_r = train_text.map(lambda row: (row[0], (row[1], row[2]))).groupByKey().mapValues(set).repartition(100)
    bur_dict = {b: {u_r[0]: u_r[1] for u_r in u_r_set} for b, u_r_set in b_u_r.collect()}

    return b_u_dict, u_b_dict, b_avg_dict, u_avg_dict, bur_dict

def process_data(train_path, test_file_name, output_file_name, spark):
    train_text=spark.textFile(train_path)
    first_train=train_text.first()
    train_text=train_text.filter(lambda row: row != first_train).map(lambda row: row.split(",")).map(lambda row: (row[1], row[0], row[2]))  
    b_u_dict, u_b_dict, b_avg_dict, u_avg_dict, bur_dict=process_traindata(train_text)   
    val_text=spark.textFile(test_file_name)
    first_val=val_text.first()
    val_text=val_text.filter(lambda row: row != first_val).map(lambda row: row.split(",")).map(lambda row: (row[1], row[0]))
    w_dict={}
    preds=[]
    for row in val_text.collect():
        pred=calcs(row[0], row[1], u_b_dict, b_u_dict, b_avg_dict, u_avg_dict, bur_dict, w_dict)
        preds.append(pred)   
    return preds

def read_u_data(spark, folder_path):
    u_text = spark.textFile(folder_path + '/user.json').map(lambda row: json.loads(row)).map(lambda x: (
        x['user_id'],
        (float(x['useful']),
         float(x['funny']),
         float(x['cool']),
         float(x['fans']),
         float(len(x['friends'])),
         float(x['average_stars']),
         float(2023 - float(x['yelping_since'][:4])),
         float(x['review_count']),
         float(x['compliment_profile']),
         float(x['compliment_more']),
         float(x['compliment_cute']),
         float(x['compliment_list']),
         float(x['compliment_note']),
         float(x['compliment_plain']),
         float(x['compliment_cool']),
         float(x['compliment_funny']),
         float(x['compliment_writer']),
         float(x['compliment_hot']),
         float(x['compliment_photos']),
         float(len(x['elite']))
         )
    ))
    u_dict = dict(u_text.collect())
    return u_dict


from datetime import datetime

def extract_hours(hours_str):
    try:
        day_hour_pairs = hours_str.split(",")
        total_hours = 0

        for day_hour_pair in day_hour_pairs:
            day, hours = day_hour_pair.split(":")
            start_hour, end_hour = map(int, hours.split("-"))
            total_hours += max(0, min(24, end_hour) - max(0, min(24, start_hour)))

        return total_hours
    except:
        return 0


def read_b_data(spark, folder_path):
  def extract_latitude(row):
        return float(row['latitude']) if 'latitude' in row and row['latitude'] is not None else 0.0

  def extract_longitude(row):
        return float(row['longitude']) if 'longitude' in row and row['longitude'] is not None else 0.0

  b_text = spark.textFile(folder_path + '/business.json').map(lambda row: json.loads(row)).map(lambda row: (
      row['business_id'],
      (
          float(row['stars']),
          float(row['review_count']),
          extract_hours(row.get('hours', '')),
          float(row['is_open']),
          # extract_city(row),
          extract_latitude(row),  
          extract_longitude(row)        
      )
  ))
  b_dict = dict(b_text.collect())
  return b_dict


def read_train_data(spark, folder_path, u_dict, b_dict):
    traintext = spark.textFile(folder_path + '/yelp_train.csv')
    first_train = traintext.first()
    traintext = traintext.filter(lambda row: row != first_train).map(lambda row: row.split(","))
    Xtrain, Ytrain = [], []
    for user, business, rating in traintext.collect():
        Ytrain.append(float(rating))
        useful, funny, cool, fans, friends, avg_stars, yelping_years, review_count, compliment_profile, \
        compliment_more, compliment_cute, compliment_list, compliment_note, compliment_plain, \
        compliment_cool, compliment_funny, compliment_writer, compliment_hot, compliment_photos, elite = u_dict.get(user, (
           None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        ))
        business_avg_stars, business_review_count, total_hours, is_open, lat, longi = b_dict.get(business, ( None, None, None, None, None, None))
        
        
        Xtrain.append([
           useful, funny, cool, fans, friends, avg_stars, yelping_years, review_count,
            compliment_profile, compliment_more, compliment_cute, compliment_list, compliment_note,
            compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_hot, compliment_photos, elite,
            business_avg_stars, business_review_count, total_hours,
            is_open, lat, longi
        ])

    Xtrain = np.array(Xtrain, dtype='float32')
    Ytrain = np.array(Ytrain, dtype='float32')
    return Xtrain, Ytrain


def read_val_data(spark, test_file_name, u_dict, b_dict):
    valtext = spark.textFile(test_file_name)
    first_val = valtext.first()
    valtext = valtext.filter(lambda row: row != first_val).map(lambda row: row.split(","))
    Xval, u_b_list = [], []
    for lines in valtext.collect():
        if len(lines) >= 2:
            user, business = lines[0], lines[1]
            useful, funny, cool, fans, friends, avg_stars, yelping_years, review_count, compliment_profile, \
            compliment_more, compliment_cute, compliment_list, compliment_note, compliment_plain, \
            compliment_cool, compliment_funny, compliment_writer, compliment_hot, compliment_photos, elite = u_dict.get(user, (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None
            ))
            business_avg_stars, business_review_count, total_hours, is_open, lat, longi = b_dict.get(business, ( None, None, None, None, None, None))
           
            

            Xval.append([
               useful, funny, cool,fans, friends, avg_stars, yelping_years, review_count,
                compliment_profile, compliment_more, compliment_cute, compliment_list, compliment_note,
                compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_hot, compliment_photos, elite,
                business_avg_stars, business_review_count, total_hours,
                is_open,lat, longi
            ])
            u_b_list.append((user, business))

    Xval = np.array(Xval, dtype='float32')
    return Xval, u_b_list


def training_model(Xtrain, Ytrain, Xval):
    paras={'reg_lambda': 96.358, 'reg_alpha': 17.636, 'colsample_bytree': 0.6909, 'subsample': 0.9, 'learning_rate': 0.0127, 'max_depth': 13,  'min_child_weight': 84, 'n_estimators': 939}
    xgbobj=XGBRegressor(**paras)
    xgbobj.fit(Xtrain, Ytrain)
    return xgbobj.predict(Xval)


def saveresults(output_file_name, u_b_list, Ypred, preds):
    alpha=0.05
    result_str="user_id, business_id, prediction\n"
    for i in range(0, len(Ypred)):
        pred=float(alpha) * float(preds[i])+(1-float(alpha)) * float(Ypred[i])
        result_str+=u_b_list[i][0]+","+u_b_list[i][1]+","+str(pred)+"\n"
    with open(output_file_name, "w") as f:
        f.writelines(result_str)


if __name__ == '__main__':
    folder_path, test_file_name, output_file_name=sys.argv[1], sys.argv[2], sys.argv[3]
    #folder_path, test_file_name, output_file_name='/content/drive/MyDrive/DSCI553/dataset','/content/drive/MyDrive/DSCI553/dataset/yelp_val_in.csv','res.csv'#sys.argv[1], sys.argv[2], sys.argv[3]
    stime=time.time()
    spark=init_spark_context("task2_3")
    spark.setLogLevel("ERROR")
    u_dict=read_u_data(spark, folder_path)
    b_dict=read_b_data(spark, folder_path)
    Xtrain, Ytrain=read_train_data(spark, folder_path,u_dict,b_dict)
    Xval, u_b_list=read_val_data(spark, test_file_name,u_dict,b_dict)
    Ypred=training_model(Xtrain, Ytrain, Xval)
    train_path=folder_path+'/yelp_train.csv'
    preds=process_data(train_path, test_file_name, output_file_name,spark)
    saveresults(output_file_name, u_b_list, Ypred, preds)
    print('Duration:', time.time()-stime)
