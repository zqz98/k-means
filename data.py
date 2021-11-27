import numpy as np

# 创建数据集
def createDataSet(num, dim, min_value=0, max_value=100):
    return np.random.randint(min_value, max_value, size=[num, dim])

if __name__ == '__main__':
    print(createDataSet(2, 2))