def neuron_output(x, w):
    if len(x) != len(w):
        raise ValueError("Кількість зв'язків і ваг не однакова")

    y = 0
    for i in range(len(x)):
        y += x[i] * w[i]

    return y

w_set = [0.5, 0.3, 1.3]

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

print("Результати для числового набору даних:")
for x_set in x_num_array:
    y = neuron_output(x_set, w_set)
    print("Вхідні дані:", x_set, "Вихідне значення нейрону:", y)

print("\nРезультати для бінарного набору даних:")
for x_set in x_bool_array:
    y = neuron_output(x_set, w_set)
    print("Вхідні дані:", x_set, "Вихідне значення нейрону:", y)
