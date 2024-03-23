from pyspark import SparkContext, SparkConf
import json, time, sys
import numpy as np
from xgboost import XGBRegressor

def load_data(spark, f_path, k_func, v_func=None):
    text=spark.textFile(f_path).map(lambda row: json.loads(row)).map(lambda row: (k_func(row), v_func(row) if v_func else row))
    return dict(text.collect())

def read_train_data(train_text, rev_dict, u_dict, b_dict):
    X, Y=[], []
    for u, b, r in train_text:
        Y.append(float(r))
        feat=[rev_dict.get(b, (0, 0, 0)), u_dict.get(u, (None, None, None)), b_dict.get(b, (None, None))]
        X.append([f for sublist in feat for f in sublist])
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

def read_val_data(val_text, rev_dict, u_dict, b_dict):
    X, u_b_list=[], []
    for u, b in val_text:
        feat=[rev_dict.get(b, (0, 0, 0)), u_dict.get(u, (None, None, None)), b_dict.get(b, (None, None))]
        X.append([f for sublist in feat for f in sublist])
        u_b_list.append((u, b))
    return np.array(X, dtype='float32'), u_b_list

def training_model(Xtrain, Ytrain, Xval):
    parameters={'reg_lambda': 0.1, 'reg_alpha': 0.01, 'colsample_bytree': 0.75, 'subsample': 0.85, 'learning_rate': 0.05, 'max_depth': 9, 'random_state': 33, 'min_child_weight': 1, 'n_estimators': 500}
    xgbobj=XGBRegressor(**parameters)
    xgbobj.fit(Xtrain, Ytrain)
    return xgbobj.predict(Xval)

def saveresults(out_path, u_b_list, Ypred):
    res_str="user_id, business_id, prediction\n"
    for (user, business), prediction in zip(u_b_list, Ypred):
        res_str += f"{user},{business},{prediction}\n"
    with open(out_path, "w") as f:
        f.writelines(res_str)


if __name__ == '__main__':
    folder_path, test_file_name, output_file_name=sys.argv[1], sys.argv[2], sys.argv[3]
    stime=time.time()
    sc=SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    rev_dict=load_data(sc, folder_path + '/review_train.json', lambda row: row['business_id'], lambda row: (float(row['useful']), float(row['funny']), float(row['cool'])))
    u_dict=load_data(sc, folder_path + '/user.json', lambda row: row['user_id'], lambda row: (float(row['average_stars']), float(row['review_count']), float(row['fans'])))
    b_dict=load_data(sc, folder_path + '/business.json', lambda row: row['business_id'], lambda row: (float(row['stars']), float(row['review_count'])))
    train_text=sc.textFile(folder_path + '/yelp_train.csv').map(lambda row: row.split(",")).collect()[1:]
    Xtrain, Ytrain=read_train_data(train_text, rev_dict, u_dict, b_dict)
    val_text=sc.textFile(test_file_name).map(lambda row: row.split(",")).collect()[1:]
    Xval, u_b_list=read_val_data(val_text, rev_dict, u_dict, b_dict)
    Ypred=training_model(Xtrain, Ytrain, Xval)
    saveresults(output_file_name, u_b_list, Ypred)
    print("Duration:", time.time() - stime)