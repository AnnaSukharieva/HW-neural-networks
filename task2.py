def neuron_output(x, w):
    return sum(x_i * w_i for x_i, w_i in zip(x, w))

def layer_output(x, weights):
    return [neuron_output(x, w) for w in weights]

def layer_output_first(x, weights):
    return list(map(lambda x_val, w: x_val * w, x, weights))

weights_layer_1 = [0.5, 0.3, 1.3]

weights_layer_2 = [
    [0.8, 0.2, 0.9],
    [1.1, 0.4, 0.7],
    [0.6, 0.5, 0.3]
]

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

def network_output(x):
    first_layer_outputs = layer_output_first(x, weights_layer_1)
    print("Вхідні дані:", x, "\nВихідні значення після першого шару:", first_layer_outputs)
    second_layer_outputs = layer_output(first_layer_outputs, weights_layer_2)
    print("Вихідні значення після другого шару:", second_layer_outputs, "\n")
    return second_layer_outputs

print("Результати для числового набору даних:")
for x_set in x_num_array:
    network_output(x_set)

print("\nРезультати для бінарного набору даних:")
for x_set in x_bool_array:
    network_output(x_set)
