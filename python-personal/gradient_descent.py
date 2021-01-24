X = [[[1,3.04] [1,3.64],[1,4.61],[1,5.57],[1,6.74], [1,7.77]]
Y = [0.94,1.01,1.09,1.11,1.20,1.30]
w = [0,0]
iteration = 0
rate = 0.01
while(iteration < 1000000):
    i = 0
    for i in range(len(X)):
        for j in range(X[i]):
            p += w[j] * X[i][j]
        delta = Y[i] - p
        for n in range(len(w)):
            pass