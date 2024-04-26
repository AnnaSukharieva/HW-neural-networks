import math

def neuron_output(x, w, S, activation_function='linear', threshold=0, k=1):
    if len(x) != len(w):
        raise ValueError("Кількість зв'язків і ваг не однакова")

    if activation_function == 'threshold':
        return 1 if S > threshold else -1, threshold
    elif activation_function == 'linear':
        return k * S
    elif activation_function == 'sigmoid':
        return 1 / (1 + (1 / math.exp(S)))
    elif activation_function == 'relu':
        return max(0, S)
    else:
        raise ValueError("Невідома функція активації")

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
    print("Вхідні дані:", x_set)

    S = sum(x_set[i] * w_set[i] for i in range(len(x_set)))
    print("S =", S)

    threshold=2
    k = -3

    y, threshold = neuron_output(x_set, w_set, S, activation_function='threshold', threshold=threshold)
    print(f"Вихідне значення нейрону (порогова функція активації з q={threshold}):", y)

    y = neuron_output(x_set, w_set, S, activation_function='linear', k=k)
    print(f"Вихідне значення нейрону (лінійна функція активації з k={k}):", y)

    y = neuron_output(x_set, w_set, S, activation_function='sigmoid')
    print("Вихідне значення нейрону (сигмоїдальна функція активації):", y)

    y = neuron_output(x_set, w_set, S,activation_function='relu')
    print("Вихідне значення нейрону (ReLU функція активації):", y, '\n')

print("\nРезультати для бінарного набору даних:")
for x_set in x_bool_array:
    print("Вхідні дані:", x_set)

    S = sum(x_set[i] * w_set[i] for i in range(len(x_set)))
    print("S =", S)

    threshold = 2
    k = 3

    y, threshold = neuron_output(x_set, w_set, S, activation_function='threshold', threshold=threshold)
    print(f"Вихідне значення нейрону (порогова функція активації з q={threshold}):", y)

    y = neuron_output(x_set, w_set, S, activation_function='linear', k=k)
    print(f"Вихідне значення нейрону (лінійна функція активації з k={k}):", y)

    y = neuron_output(x_set, w_set, S, activation_function='sigmoid')
    print("Вихідне значення нейрону (сигмоїдальна функція активації):", y)

    y = neuron_output(x_set, w_set, S, activation_function='relu')
    print("Вихідне значення нейрону (ReLU функція активації):", y, '\n')
