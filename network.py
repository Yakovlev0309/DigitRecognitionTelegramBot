import numpy as np
import csv

model_path = "models/model.npz"

topology = [784, 10, 15, 15]
input_count = topology[0]
output_count = topology[1]
h_count = topology[2]
h2_count = topology[3]

rate = 0.001
epoch_count = 10

w1 = np.matrix(2 * np.random.random_sample((input_count, h_count)) - 1)
b1 = np.matrix(2 * np.random.random_sample((1, h_count)) - 1)
w2 = np.matrix(2 * np.random.random_sample((h_count, h2_count)) - 1)
b2 = np.matrix(2 * np.random.random_sample((1, h2_count)) - 1)
w3 = np.matrix(2 * np.random.random_sample((h2_count, output_count)) - 1)
b3 = np.matrix(2 * np.random.random_sample((1, output_count)) - 1)


def train_network(train_name, train_count=60000):
    csv_train = mnist_csv_read(train_name, train_count)

    global w1, w2, w3, b1, b2, b3

    print("training started")

    for epoch in range(epoch_count):
        print("epoch", epoch)
        train(csv_train)  # Обучение
        np.random.shuffle(csv_train)

    print("training completed")
    np.savez(model_path, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)


def run(data_name, weights_name):
    weights = np.load(weights_name)
    w_1, b_1, w_2, b_2, w_3, b_3 = np.matrix(weights['w1']), np.matrix(weights['b1']), np.matrix(weights['w2']), \
        np.matrix(weights['b2']), np.matrix(weights['w3']), np.matrix(weights['b3'])

    data = csv_read(data_name)
    x = np.transpose(data)

    # Прямое распространение
    t1 = np.dot(x.T, w_1) + b_1
    h1 = bipolar_sigmoid(t1)

    t2 = np.dot(h1, w_2) + b_2
    h2 = bipolar_sigmoid(t2)

    t3 = np.dot(h2, w_3) + b_3
    z = softmax(t3)

    return np.argmax(z)


def train(csv_train):
    global w1, b1, w2, b2, w3, b3
    row_count, column_count = csv_train.shape

    for r_i in range(row_count):
        tmp = csv_train[r_i]
        tmp = np.transpose(tmp)
        x = tmp[1:]
        number = tmp[0][0]
        y = np.matrix(np.zeros((output_count, 1)))
        y[int(number)][0] = 1
        y = y.reshape(1, output_count)

        # Прямое распространение
        x = x / 255

        t1 = np.dot(x.T, w1) + b1
        h1 = bipolar_sigmoid(t1)

        t2 = np.dot(h1, w2) + b2
        h2 = bipolar_sigmoid(t2)

        t3 = np.dot(h2, w3) + b3
        z = softmax(t3)

        # Обратное распространение

        de_dt3 = z - y
        de_dw3 = np.dot(h2.T, de_dt3)
        de_db3 = de_dt3

        de_dh2 = np.dot(de_dt3, w3.T)
        de_dt2 = np.multiply(de_dh2, d_bipolar_sigmoid(t2))
        de_dw2 = np.dot(h1.T, de_dt2)
        de_db2 = de_dt2

        de_dh1 = np.dot(de_dt2, w2.T)
        de_dt1 = np.multiply(de_dh1, d_bipolar_sigmoid(t1))
        de_dw1 = np.dot(x, de_dt1)
        de_db1 = de_dt1

        w1 = w1 - rate * de_dw1
        b1 = b1 - rate * de_db1
        w2 = w2 - rate * de_dw2
        b2 = b2 - rate * de_db2
        w3 = w3 - rate * de_dw3
        b3 = b3 - rate * de_db3


def d_bipolar_sigmoid(x):
    return 2 * np.exp(x) / (np.exp(2 * x) + 2 * np.exp(x) + 1)


def bipolar_sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1


def softmax(t):
    return np.exp(t) / np.sum(np.exp(t))


# Функция потерь кросс-энтропии
def get_e(y, z):
    z = z.transpose()
    return -np.sum(y * np.log(z))


def mnist_csv_read(name, row_count):
    reader = csv.reader(open(name, newline=''))
    x = np.matrix(np.zeros((row_count, input_count + 1)))
    index = 0
    for row in reader:
        if index == row_count:
            break
        x[index] = row
        index = index + 1
    return x


def csv_read(name):
    data = np.genfromtxt(name, delimiter=',')
    flat_data = data.flatten()

    return flat_data
