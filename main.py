# python
# author:张辉
# time:2020/10/25
# python
# author:张辉
# time:2020/10/22
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
random.seed(3)
iris = []
train_set = []
test_set = []
valida_set = []

with open("iris.txt")as file_object:
    for line in file_object:
        line = line.strip('\n')
        words = line.split(",")
        iris.append(words)
iris.pop()
random.shuffle(iris)

# 数据集划分
for i in range(150):
    for j in range(5):
        if iris[i][j] == 'Iris-setosa':
            train_set.append(0)
        elif iris[i][j] == 'Iris-versicolor':
            train_set.append(1)
        elif iris[i][j] == 'Iris-virginica':
            train_set.append(2)
        else:
            train_set.append(float(iris[i][j]))

train_set = torch.Tensor(train_set).reshape(150, 5)
x, y = torch.split(train_set, 4, 1)
y = y.long()
train_set_x, val_x = torch.split(x, 120, 0)
valida_set_x, test_set_x = torch.split(val_x, 15, 0)
train_set_y, val_y = torch.split(y, 120, 0)
valida_set_y, test_set_y = torch.split(val_y, 15, 0)
# 将数据集转换为列向量
train_set_x = train_set_x.transpose(1, 0)
# train_set_y = train_set_y.transpose(1, 0)
valida_set_x = valida_set_x.transpose(1, 0)
# valida_set_y = valida_set_y.transpose(1, 0)
test_set_x = test_set_x.transpose(1, 0)
# test_set_y = test_set_y.transpose(1, 0)


# 初始化参数
def initialize_parameters(n_feature, n_hidden1, n_hidden2, n_output):
    w1 = torch.tensor([[0.278, 0.779, 0.130, 0.115],
                       [0.393, 0.536, 0.781, 0.271],
                       [0.493, 0.692, 0.389, 0.515],
                       [0.101, 0.408, 0.304, 0.620]])
    w2 = torch.tensor([[0.173, 0.523, 0.418, 0.766],
                       [0.467, 0.701, 0.437, 0.578],
                       [0.670, 0.616, 0.710, 0.660],
                       [0.936, 0.809, 0.513, 0.378]])
    w3 = torch.tensor([[0.461, 0.679, 0.220, 0.463],
                       [0.286, 0.502, 0.598, 0.576],
                       [0.311, 0.122, 0.393, 0.673]])
    parameters = {"w1": w1, "w2": w2, "w3": w3}
    return parameters


# 前向传播
def forward(x, parameters):
    # 正向传播过程
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    z1 = torch.mm(w1, x)
    a1 = torch.sigmoid(z1)
    z2 = torch.mm(w2, a1)
    a2 = torch.sigmoid(z2)
    z3 = torch.mm(w3, a2)
    a3 = torch.softmax(z3, 0)
    cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}
    return a3, cache


# 计算代价函数
def compute_cost(a3, y, parameters):
    # 样本总数
    m = y.shape[1]
    # 取参数
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    cost = torch.zeros(m, 1)
    # 计算交叉熵
    for item in range(m):
        cost[item][0] = - np.log(a3[y[item][0]][item])
    loss = cost.sum() / m
    return loss


# 反向传播
def backward_propagation(parameters, cache, x, y, print_dw=False):
    # 训练样本数
    m = x.shape[1]
    # 取参数
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    # 取网络层输出值
    a1 = cache['a1']
    a2 = cache['a2']
    a3 = cache['a3']
    z1 = cache['z1']
    z2 = cache['z2']
    z3 = cache['z3']
    # backward
    da3 = a3.clone()
    for i in range(m):
        da3[y[i][0]][i] = a3[y[i][0]][i] - 1
    dw1 = torch.zeros(4, 4)
    dw2 = torch.zeros(4, 4)
    dw3 = torch.zeros(3, 4)
    dz1 = torch.mul(a1, 1 - a1)
    dz2 = torch.mul(a2, 1 - a2)
    for i in range(m):
        da3_x = da3[:, i].reshape(3, 1)
        a2_x = a2.transpose(1, 0)[i, :].reshape(1, 4)
        dw3 = torch.mm(da3_x, a2_x) + dw3
        w3_da3 = torch.mm(w3.transpose(1, 0), da3_x)
        s2 = dz2[:, i].reshape(4, 1)
        s2_w3_da3 = torch.mul(w3_da3, s2)
        a1_x = a1.transpose(1, 0)[i, :].reshape(1, 4)
        dw2 = torch.mm(s2_w3_da3, a1_x) + dw2
        w2_s2_w3_da3 = torch.mm(w2.transpose(1, 0), s2_w3_da3)
        s1 = dz1[:, i].reshape(4, 1)
        s1_w2_s2_w3_da3 = torch.mul(w2_s2_w3_da3, s1)
        x1 = train_set_x.transpose(1, 0)[i, :].reshape(1, 4)
        dw1 = torch.mm(s1_w2_s2_w3_da3, x1) + dw1
    dw1 = dw1 / m
    dw2 = dw2 / m
    dw3 = dw3 / m
    grads = {"dw1": dw1, "dw2": dw2, "dw3": dw3}
    if print_dw:
        print("手动求导", grads['dw1'])
        print("手动求导", grads['dw2'])
        print("手动求导", grads['dw3'])
    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate=3):
    # 取参数
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    # 取导数
    dw1 = grads["dw1"]
    dw2 = grads["dw2"]
    dw3 = grads["dw3"]
    # 更新参数
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    w3 = w3 - learning_rate * dw3

    parameters = {"w1": w1, "w2": w2, "w3": w3}
    return parameters


# 构建神经网络
def nn_module(x, y, num_itertions=1000, print_cost=False, print_dw=False, print_loss=False):
    # 初始化参数
    # n_h 为隐藏层节点数
    parameters = initialize_parameters(4, 4, 4, 3)
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    m = x.shape[1]
    # 图像横纵坐标
    px = []
    py = []
    # 训练迭代次数
    for i in range(0, num_itertions):
        # forward
        a3, cache = forward(x, parameters)
        # cost
        cost = compute_cost(a3, y, parameters)
        # back
        grads = backward_propagation(parameters, cache, x, y, print_dw)
        # gradient
        parameters = update_parameters(parameters, grads)
        if print_cost:
            print("cost", cost)
        # print("a3", a3)
        # 画loss曲线图
        px.append(i)
        py.append(cost)
    plt.plot(px, py)
    plt.xlabel('step')
    plt.ylabel('Loss')
    if print_loss:
        plt.show()

    # 计算准确率
    y_pred_max, index = a3.max(0)
    current = 0
    for i in range(m):
        if index[i] == y[i][0]:
            current += 1
    print(current / m)
    return parameters


# 训练集结果
train_parameters = nn_module(train_set_x, train_set_y, num_itertions=320, print_cost=True, print_loss=True)
train_a3, train_cache = forward(train_set_x, train_parameters)
# 验证集结果
valida = nn_module(valida_set_x, valida_set_y, num_itertions=1, print_cost=True)
# 测试集结果
test_a3, test_cache = forward(test_set_x, train_parameters)
# 计算测试集准确率
y_pred_max, index = test_a3.max(0)
test_current = 0
for i in range(15):
    if index[i] == test_set_y[i][0]:
        test_current += 1
print("test_current", test_current / 15)


# 自动求导
for item in range(1):
    # 前向传播
    w1 = torch.tensor([[0.278, 0.779, 0.130, 0.115],
                       [0.393, 0.536, 0.781, 0.271],
                       [0.493, 0.692, 0.389, 0.515],
                       [0.101, 0.408, 0.304, 0.620]], requires_grad=True)
    w2 = torch.tensor([[0.173, 0.523, 0.418, 0.766],
                       [0.467, 0.701, 0.437, 0.578],
                       [0.670, 0.616, 0.710, 0.660],
                       [0.936, 0.809, 0.513, 0.378]], requires_grad=True)
    w3 = torch.tensor([[0.461, 0.679, 0.220, 0.463],
                       [0.286, 0.502, 0.598, 0.576],
                       [0.311, 0.122, 0.393, 0.673]], requires_grad=True)
    auto_h1 = torch.mm(w1, train_set_x)
    auto_a1 = torch.sigmoid(auto_h1)
    auto_h2 = torch.mm(w2, auto_a1)
    auto_a2 = torch.sigmoid(auto_h2)
    auto_h3 = torch.mm(w3, auto_a2)
    y_pred = torch.softmax(auto_h3, 0)
    auto_cost = torch.zeros(120, 1)
    # compute loss
    for i in range(120):
        auto_cost[i][0] = - torch.log(y_pred[train_set_y[i][0]][i])
    loss = auto_cost.sum() / 120
    loss.backward()
    print("自动求导", w1.grad)
    print("自动求导", w2.grad)
    print("自动求导", w3.grad)
# 与手动求导比较
train_parameters = nn_module(train_set_x, train_set_y, num_itertions=1, print_cost=False, print_dw=True)

