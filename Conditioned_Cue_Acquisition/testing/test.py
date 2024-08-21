keys = ['A', 'B', 'C']
vars = [0, 1, 1]

for i, tup in enumerate(zip(vars, keys)):
    v, k = tup
    print(i, v, k)

for i in range(50):
    print(i//2)
