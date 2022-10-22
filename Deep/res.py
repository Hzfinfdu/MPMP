import numpy as np

loss = []
with open('res.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        data = line.strip().split(' ')
        loss.append(float(data[-1]))
        if i % 5 == 4:
            loss = np.array(loss)
            print(f'{data[1]}, {round(loss.mean() * 100, 2)}({round(loss.std() * 80, 2)})')
            loss = []

