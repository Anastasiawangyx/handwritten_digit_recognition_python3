import random
import numpy as np

'''
定义一个神经网络类
'''
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]  # randn返回的是一个0，1之间的随机小数，格式是y行1列。
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        '''
        :param a: 神经网络的输入
        :return a: 输出层
        '''
        for b, w in zip(self.bias, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        随机梯度下降算法，将训练数据分成相同大小的mini_batch，在每个batch不断迭代得到权重和偏置。
        :param training_data: 训练数据
        :param epochs: 迭代周期
        :param mini_batch_size: 一个mini_batch的大小
        :param eta: 学习速率
        :param test_data: 测试数据
        :return: None
        '''
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0} :{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete.".format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        在每一个mini_batch内的每一个数据使用反向传播算法，得到权重和偏置的在一个mini_batch内的更新。
        :param mini_batch: 一个用来训练网络的数据batch，由测试数据打乱后分割而成。
        :param eta: 学习速率
        :return: None
        '''
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.bias = [b - (eta / len(mini_batch) * nb) for b, nb in zip(self.bias, nabla_b)]
        self.weights = [w - (eta / len(mini_batch) * nw) for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        '''
        反向传播算法，利用四个公式，得到一个单独数据输入时，权重和偏置的改变
        :param x: 输入的数据
        :param y: 期望输出
        :return:
        '''
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        '''
        得到正确分类的个数
        :param test_data: 测试数据
        :return:
        '''
        test_result = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_result)

    def cost_derivative(self,output_activations,y):
        '''
        得到代价函数的导数
        :param output_activations: 输出激活值
        :param y: 期望输出
        :return:
        '''
        return output_activations-y

def sigmoid(z):
    '''
    S型神经元的格式
    :param z: 中间值
    :return:
    '''
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    '''
    S型神经元的导数
    :param z:
    :return:
    '''
    return sigmoid(z) * (1 - sigmoid(z))
