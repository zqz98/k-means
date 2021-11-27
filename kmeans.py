import random
import numpy as np
from tqdm import tqdm

# 计算欧氏距离
def calcDis_e(dataset, centroids, k):
    batch, dim = dataset.shape
    dataset = dataset.repeat(k, axis=0).reshape(-1, dim)
    centroids = np.expand_dims(centroids, 0).repeat(batch, axis=0).reshape(-1, dim)
    clalist = np.sum((dataset - centroids) ** 2, axis=1).reshape(-1, k)
    return clalist

# 计算cos距离
def calcDis_cos(dataset, centroids, k):
    dataSet_norm = np.expand_dims(np.linalg.norm(dataset, axis=1), 1)
    centroids_norm = np.expand_dims(np.linalg.norm(centroids, axis=1), 0)
    clalist = np.dot(dataset, centroids.T) / (dataSet_norm * centroids_norm)
    return 1 - clalist

clacdis_dir = {'calcDis_e':calcDis_e, 'calcDis_cos':calcDis_cos}

# 选择距离函数
def clacdis_choose(distance):
    return clacdis_dir.get(distance, 'calcDis_e')


# 计算中心
def center(dataset, min_dist_indices, k):
    new_centroids = []
    for k_class in range(k):
        new_center = dataset[min_dist_indices == k_class]
        new_center = np.mean(new_center, 0)
        new_centroids.append(new_center)
    return np.array(new_centroids)


# 计算质心
def classify(dataset, centroids, k, distance):
    # 计算样本到质心的距离
    clacdis = clacdis_choose(distance)
    clalist = clacdis(dataset, centroids, k)
    # 分组并计算新的质心
    min_dist_indices = np.argmin(clalist, axis=1)
    new_centroids = center(dataset, min_dist_indices, k)
    # 计算变化量
    changed = new_centroids - centroids

    return changed, new_centroids


# k-means算法
def kmeans(dataset, k, iteration, error, distance):
    # 随机取质心
    centroids = random.sample(dataset.tolist(), k)
    centroids = np.array(centroids)
    # 更新质心 直到变化量小于error，或者达到最大迭代次数
    changed, new_centroids = classify(dataset, centroids, k, distance)
    for _ in tqdm(range(iteration)):
        if np.linalg.norm(changed) < error:
            print('The change value is %6f, which is less than the threshold and the iteration is terminated' % (np.linalg.norm(changed)))
            break
        changed, new_centroids = classify(dataset, new_centroids, k, distance)
    centroids = np.sort(new_centroids, 0)

    # 根据质心计算每个聚类
    cluster = []
    # 选择距离函数
    clacdis = clacdis_choose(distance)
    clalist = clacdis(dataset, centroids, k)
    min_dist_indices = np.argmin(clalist, axis=1)
    for k_class in range(k):
        cluster.append(dataset[min_dist_indices == k_class])

    return centroids, cluster