from pyspark import SparkContext
import time, math, csv, sys

def pearson(neigh, u_set, u_dict, b_avg, n_avg, b_rate):
    u_n_set, u_n_dict=set(b_rate.get(neigh, {}).keys()), dict(b_rate.get(neigh, {}))
    commons=u_set.intersection(u_n_set)
    if not commons: return float(b_avg / n_avg)
    num, denom_b, denom_n=0, 0, 0
    for user in commons:
        n_b, n_n=u_dict.get(user, 0) - b_avg, u_n_dict.get(user, 0) - n_avg
        num += n_b * n_n
        denom_b += n_b * n_b
        denom_n += n_n * n_n
    denom=math.sqrt(denom_b * denom_n)
    if denom == 0: return 1 if num == 0 else -1
    return num / denom

def task2_1(train, test, output):
    def rate_no_biz(user):
        return str(u_avg.get(user, 2.5))

    def rating_pred(weight):
        n=min(len(weight), 30)
        weight.sort(key=lambda x: x[0], reverse=True)
        weight=weight[:n]
        sum_similarity=0
        sum_weight=0
        for i in range(n):
            similarity=weight[i][0]
            rate=weight[i][1]
            sum_similarity += abs(similarity)
            sum_weight += similarity * rate
        return sum_weight / sum_similarity

    def item_based_collaborative_filter(line):
        u, b=line[0], line[1]
        b_avg=b_avg_dict.get(b, 2.5)
        u_set=set(b_rate.get(b, {}).keys())
        u_dict=dict(b_rate.get(b, {}))
        weight=[]
        for neighbor in u_rate.get(u, {}).keys():
            if neighbor != b:
                cur_neighbor_rate=b_rate.get(neighbor, {}).get(u)
                neighbor_rating_avg=b_avg_dict.get(neighbor, 2.5)
                pearson_coef=pearson(neighbor, u_set, u_dict, b_avg, neighbor_rating_avg, b_rate)
                if pearson_coef > 1:
                    pearson_coef=1 / pearson_coef
                if pearson_coef > 0:
                    weight.append((pearson_coef, cur_neighbor_rate))

        return u, b, rating_pred(weight)

    start=time.time()
    sc=SparkContext.getOrCreate()

    train_rdd, test_rdd=sc.textFile(train), sc.textFile(test)
    train_header, test_header=train_rdd.first(), test_rdd.first()
    train_data, test_data=train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(',')), test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

    u_rate=train_data.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    b_rate=train_data.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    u_avg=train_data.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()
    b_avg_dict=train_data.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

    prediction=test_data.sortBy(lambda x: (x[0], x[1])).map(item_based_collaborative_filter).collect()

    with open(output, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for i in range(len(prediction)):
            f.write(f"{prediction[i][0]},{prediction[i][1]},{prediction[i][2]}\n")

    print(time.time() - start)

train_file_name, test_file_name, output_file_name=sys.argv[1],sys.argv[2],sys.argv[3]
task2_1(train_file_name, test_file_name, output_file_name)
