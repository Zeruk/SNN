import snn
import numpy as np

options = ('solid', 'horisontal line', 'vertical line', 'diagonal line', 'dot')

def variant(data):
    arr_sum = data.sum()
    if arr_sum == 0 or arr_sum == 4:
        return 0
    elif arr_sum == 3 or arr_sum == 1:
        return 4
    elif arr_sum == 2 and data[0, 0] == data[1, 0]:
        return 1
    elif arr_sum == 2 and data[0, 0] == data[2, 0]:
        return 2
    return 3

def pattern(data):
    index = variant(data)
    result = np.zeros(shape=(5, 1), dtype=float)
    result[index, 0] = 1
    return result

# our neural net
nn = snn.perceptron(4, 10, 5)

# train it
for i in range(300000):
    # input
    data = np.random.random(size=(4, 1))
    nn.input(data)

    # tweak nn
    error = nn.tweak(pattern(data))

    # input error
    if i % 10000 == 0:
        print('#{0} error - {1}'.format(i, error))

# test it
for i in range(5):
    print('Input vector')
    data = np.array(list(map(int, input().split())))[:, np.newaxis]
    print(data.reshape((2, 2)))
    answer = nn.input(data)
    print(options[variant(data)])
