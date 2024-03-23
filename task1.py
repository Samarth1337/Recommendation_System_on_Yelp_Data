from pyspark import SparkContext
import time, itertools, random, sys

H,B,R=50,25,2
a_v,b_v,c_v=random.sample(range(1, 1000), H), random.sample(range(1, 1000), H), random.sample(range(1, 1000), H)

def hash_fn(x, a, b, c, num):
    res=-1
    for u in x:
        if res == -1:
            res=(a + (u * b) + c) % num
        else:
            res=min(res, (a + (u * b) + c) % num)
    return res

def band_fn(x):
    res=[]
    for i in range(B):
        res.append(((i, tuple(x[1][i * R: (i + 1) * R])), [x[0]]))
    return res

def sort_fn(pair):
    lst=list(pair)
    lst.sort()
    return tuple(lst)

def jaccard_sim(business, char_matrix):
    set1, set2=set(char_matrix[business[0]]), set(char_matrix[business[1]])
    sim=len(set1.intersection(set2)) / len(set1.union(set2))
    return business[0], business[1], sim

def main(input_file, output_file):
    s=time.time()
    sc=SparkContext.getOrCreate()
    rdd= sc.textFile(input_file)
    h=rdd.first()
    data=rdd.filter(lambda x: x != h).map(lambda x: x.split(',')).map(lambda x: (x[1], x[0]))
    u_r=data.map(lambda x: x[1]).distinct().zipWithIndex()
    u_n, u_d=u_r.count(), u_r.collectAsMap()

    m=data.map(lambda x: (x[0], u_d[x[1]])).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
    c_m, s_m=m.collectAsMap(), m.map(lambda x: (x[0], [hash_fn(x[1], a_v[i], b_v[i], c_v[i], u_n) for i in range(H)]))
    c_p=s_m.flatMap(lambda x: band_fn(x)).reduceByKey(lambda x, y: x + y).reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1).flatMap(lambda p: list(itertools.combinations(p[1], 2))).map(lambda p: sort_fn(p)).distinct()
    f_r=c_p.map(lambda b: jaccard_sim(b, c_m)).filter(lambda x: x[2] >= 0.5).sortBy(lambda x: (x[0], x[1]))

    with open(output_file, 'w') as f:
        f.write("business_id_1, business_id_2, similarity\n")
        for l in f_r.collect():
            f.write(f"{l[0]},{l[1]},{l[2]}\n")

    print("Duration: ", time.time() - s)

if __name__ == "__main__":
    input_file_name, output_file_name=sys.argv[1],sys.argv[2]
    main(input_file_name, output_file_name)
