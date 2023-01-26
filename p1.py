import numpy as np


def act_func(a):
    return np.where(a > 0, 1, 0)


def binary_to_decimal(x):
    return int(''.join(str(i) for i in x), 2)


def rnn_addition(x):
    U = np.asarray([(1, 1), (1, 1), (1, 1)])
    W = np.asarray([(0, 1, 0), (0, 1, 0), (0, 1, 0)])

    if sum(x[0]) == 0:
        h_0 = np.asarray(([0, 0, 0]))
    elif sum(x[0]) == 1:
        h_0 = np.asarray(([1, 0, 0]))
    else:
        h_0 = np.asarray(([1, 1, 0]))

    b_h = np.asarray([0, -1, -2])

    v = np.asarray([1, -1, 1])
    b_y = 0

    y_0 = act_func(np.dot(v.T, h_0) + b_y)

    h = [h_0]
    y = [y_0]
    for t in range(1, len(x)):
        h_t = act_func(np.dot(U, x[t]) + np.dot(W, h[t-1]) + b_h)
        h.append(h_t)
        y_t = act_func(np.dot(v.T, h_t) + b_y)
        y.append(y_t)

    return binary_to_decimal(y[::-1])


def main():
    num1 = int(input('Enter first number: '))
    num2 = int(input('Enter second number: '))

    # convert numbers to binary
    num1 = np.binary_repr(num1)
    num2 = np.binary_repr(num2)

    # pad the largest number with a zero on the left
    num1 = num1.zfill(len(num1) + 1)

    # pad numbers with zeros to make them the same length
    max_len = max(len(num1), len(num2))
    num1 = num1.zfill(max_len)
    num2 = num2.zfill(max_len)

    # reverse the strings
    num1 = num1[::-1]
    num2 = num2[::-1]

    # convert strings to lists of integers
    num1 = list(map(int, num1))
    num2 = list(map(int, num2))

    # create a list of tuples
    x = list(zip(num1, num2))

    # compute the sum
    ans = rnn_addition(np.asarray(x))

    print('The sum is: {}'.format(ans))


if __name__ == '__main__':
    main()