from pyspark import SparkContext, SparkConf
import json, time, sys
import numpy as np
from xgboost import XGBRegressor

def init_spark_context(app_name):
    return SparkContext(appName=app_name)

def calcs(b, u, u_b_dict, b_u_dict, b_avg_dict, u_avg_dict, bur_dict, w_dict):
    if u not in u_b_dict:
        return 3.5
    if b not in b_u_dict:
        return u_avg_dict[u]
    w_list=[]
    for b1 in u_b_dict[u]:
        temp=tuple(sorted((b1, b)))
        w=w_dict.get(temp)
        if w is None:
            w=calculate_weight(b, b1, u, bur_dict, b_avg_dict, b_u_dict)
            w_dict[temp]=w
        w_list.append((w, float(bur_dict[b1][u])))
    return calcr(w_list)

def calculate_weight(b, b1, u, bur_dict, b_avg_dict, b_u_dict):
    u_inter=b_u_dict[b] & b_u_dict[b1]
    if len(u_inter) <= 1:
        return (5.0-abs(b_avg_dict[b]-b_avg_dict[b1]))/5
    return calculate_weight_with_interactions(b, b1, u, u_inter, bur_dict)

def calculate_weight_with_interactions(b, b1, u, u_inter, bur_dict):
    u_inter=list(u_inter)
    weights=[(5.0-abs(float(bur_dict[b][u_inter[i]])-float(bur_dict[b1][u_inter[i]]))/5) for i in range(2)]
    return sum(weights)/2

def calcr(w_list):
    w_list_can=sorted(w_list, key=lambda x: -x[0])[:15]
    X=sum(w * r for w, r in w_list_can)
    Y=sum(abs(w) for w, _ in w_list_can)
    return X/Y if Y != 0 else 3.5

def process_traindata(train_text):
    b_u_train=train_text.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set)
    b_u_dict=dict(b_u_train.collect())    
    u_b_train=train_text.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)
    u_b_dict=dict(u_b_train.collect())   
    b_avg=train_text.map(lambda row: (row[0], float(row[2]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1])/len(x[1])))
    b_avg_dict=dict(b_avg.collect())
    u_avg=train_text.map(lambda row: (row[1], float(row[2]))).groupByKey().mapValues(list).map(lambda x: (x[0], sum(x[1])/len(x[1])))
    u_avg_dict=dict(u_avg.collect())
    b_u_r=train_text.map(lambda row: (row[0], (row[1], row[2]))).groupByKey().mapValues(set)
    bur_dict={b: {u_r[0]: u_r[1] for u_r in u_r_set} for b, u_r_set in b_u_r.collect()}
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

def read_rev_data(spark, folder_path):
    review_lines=spark.textFile(folder_path+'/review_train.json').map(lambda row: json.loads(row)).map(lambda row: (row['business_id'], (float(row['useful']), float(row['funny']), float(row['cool'])))).groupByKey().mapValues(list)
    rev_dict={}
    for business, items in review_lines.collect():
        useful_total, funny_total, cool_total=0, 0, 0
        for item in items:
            item_list=list(item)
            useful_total+=item_list[0]
            funny_total+=item_list[1]
            cool_total+=item_list[2]

        for business in rev_dict:
            num_items=len(items)
            useful_avg=useful_total/num_items
            funny_avg=funny_total/num_items
            cool_avg=cool_total/num_items
            rev_dict[business]=(useful_avg, funny_avg, cool_avg)
    return rev_dict

def read_u_data(spark, folder_path):
    u_text=spark.textFile(folder_path+'/user.json').map(lambda row: json.loads(row)).map(lambda row: (row['user_id'], (float(row['average_stars']), float(row['review_count']), float(row['fans']))))
    u_dict=dict(u_text.collect())
    return u_dict

def read_b_data(spark, folder_path):
    b_text=spark.textFile(folder_path+'/business.json').map(lambda row: json.loads(row)).map(lambda row: (row['business_id'], (float(row['stars']), float(row['review_count']))))
    b_dict=dict(b_text.collect())
    return b_dict

def read_train_data(spark, folder_path):
    traintext=spark.textFile(folder_path+'/yelp_train.csv')
    first_train=traintext.first()
    traintext=traintext.filter(lambda row: row != first_train).map(lambda row: row.split(","))
    Xtrain, Ytrain=[], []
    for user, business, rating in traintext.collect():
        Ytrain.append(rating)
        useful, funny, cool=None, None, None
        if business in rev_dict:
            useful, funny, cool=rev_dict[business]
        user_avg_stars, user_review_count, user_fans=u_dict.get(user, (None, None, None))
        business_avg_stars, business_review_count=b_dict.get(business, (None, None))
        Xtrain.append([useful, funny, cool, user_avg_stars, user_review_count, user_fans, business_avg_stars, business_review_count])
    Xtrain=np.array(Xtrain, dtype='float32')
    Ytrain=np.array(Ytrain, dtype='float32')
    return Xtrain, Ytrain

def read_val_data(spark, test_file_name):
    valtext=spark.textFile(test_file_name)
    first_val=valtext.first()
    valtext=valtext.filter(lambda row: row != first_val).map(lambda row: row.split(","))
    Xval, u_b_list=[], []
    for lines in valtext.collect():
        if len(lines) >= 2:
            user, business=lines[0],lines[1]
            useful, funny, cool=None, None, None
            if business in rev_dict:
                useful, funny, cool=rev_dict[business]
            user_avg_stars, user_review_count, user_fans=u_dict.get(user, (None, None, None))
            business_avg_stars, business_review_count=b_dict.get(business, (None, None))
            Xval.append([useful, funny, cool, user_avg_stars, user_review_count, user_fans, business_avg_stars, business_review_count])
            u_b_list.append((user, business))
    Xval=np.array(Xval, dtype='float32')
    return Xval, u_b_list

def training_model(Xtrain, Ytrain, Xval):
    paras={'reg_lambda': 0.1, 'reg_alpha': 0.01, 'colsample_bytree': 0.75, 'subsample': 0.85, 'learning_rate': 0.05, 'max_depth': 9, 'random_state': 33, 'min_child_weight': 1, 'n_estimators': 500}
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
    stime=time.time()
    spark=init_spark_context("task2_3")
    rev_dict=read_rev_data(spark, folder_path)
    u_dict=read_u_data(spark, folder_path)
    b_dict=read_b_data(spark, folder_path)
    Xtrain, Ytrain=read_train_data(spark, folder_path)
    Xval, u_b_list=read_val_data(spark, test_file_name)
    Ypred=training_model(Xtrain, Ytrain, Xval)
    train_path=folder_path+'/yelp_train.csv'
    preds=process_data(train_path, test_file_name, output_file_name,spark)
    saveresults(output_file_name, u_b_list, Ypred, preds)
    print('Duration:', time.time()-stime)
