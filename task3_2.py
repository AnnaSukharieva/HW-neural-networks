import numpy as np

k = 0.3
q = 1.5
def sigmoid(x):
    return 1 / (1 + (1 / np.exp(x)))

def threshold(x):
    if x >= q:
        return 1
    else:
        return -1

def ReLU(x):
    return np.maximum(0, x)

def linear(x):
    return k*x

def neuron_output(x, w, activation_func):
    return activation_func(np.dot(x, w))

def layer_output(x, weights, activation_func):
    return [neuron_output(x, w, activation_func) for w in weights]

def layer_output_first(x, weights, activation_func):
    return [activation_func(x_val * w) for x_val, w in zip(x, weights)]

weights_layer_1 = [0.5, 0.3, 1.3]
weights_layer_2 = [
    [0.8, 0.2, 0.9],
    [1.1, 0.4, 0.7],
    [0.6, 0.5, 0.3]
]

activation_functions = [linear, sigmoid, ReLU, threshold]

x_num_array = [
    [-6, 2, 1],
    [6, 4, 4],
    [-8, -3, 6],
    [3, 8, 1]
]

x_bool_array = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]

def network_output(x, activation_func_first, activation_func_second):
    first_layer_outputs = layer_output_first(x, weights_layer_1, activation_func_first)
    print("k:", k, "q:", q)
    print("Вхідні дані:", x, "\nВихідні значення після першого шару:", first_layer_outputs)
    second_layer_outputs = layer_output(first_layer_outputs, weights_layer_2, activation_func_second)
    print("Вихідні значення після другого шару:", second_layer_outputs, "\n")
    return second_layer_outputs


activation_combinations = [
    (linear, linear),
    (sigmoid, threshold),
    (ReLU, linear),
    (threshold, ReLU)
]
print("Результати для числового набору даних:")
for activation_func_first, activation_func_second in activation_combinations:
    print(f"Перший шар: {activation_func_first.__name__}, Другий шар: {activation_func_second.__name__}")
    for x_set in x_num_array:
        network_output(x_set, activation_func_first, activation_func_second)

print("\nРезультати для бінарного набору даних:")
for activation_func_first, activation_func_second in activation_combinations:
    print(f"Перший шар: {activation_func_first.__name__}, Другий шар: {activation_func_second.__name__}")
    for x_set in x_bool_array:
        network_output(x_set, activation_func_first, activation_func_second)
