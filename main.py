import random
import numpy as np
from kmeans import kmeans
from options import args
from data import createDataSet
from kmeans import clacdis_choose


def train():
    dataset = createDataSet(args.num, args.dim, args.min, args.max)
    centroids, cluster = kmeans(dataset, args.k, args.iteration, args.error, args.distance)
    print('质心为：%s' % centroids)
    print('聚类为：%s' % cluster)
    return centroids

def test(centroids):
    test_data = createDataSet(1, args.dim, args.min, args.max)
    clacdis = clacdis_choose(args.distance)
    clalist = clacdis(test_data, centroids, args.k)
    min_dist_indices = np.argmin(clalist, axis=1)
    print('测试数据为: %s,  所属类别为：%s, 质心为：%s' % (test_data, min_dist_indices, centroids[min_dist_indices]))


def main():
    print(args)
    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    centroids = train()
    test(centroids)

if __name__ == '__main__':
    main()