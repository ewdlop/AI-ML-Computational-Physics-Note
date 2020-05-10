def linear_classifier(c1,c2, iteration, threshold):
    w = [1, 0, 0]
    total = len(c1) + len(c2)
    correct = 0
    step = 0.6
    iteration = 0
    while(correct < total):
        iteration += 1
        if(iteration >= threshold):
            print("The dataset might be not linear separable")
            break
        for x in c1:
            wx = 0
            i = 0
            j = 0
            for xi in x:
                if(i < len(w)):
                    wx += xi * w[i]
                    i += 1
            if (wx <= 0):
                for wj in w:
                    if(j < len(w)):
                        w[j] += step * x[j]
                        j+=1
                correct = 1
            else:
                correct += 1
            if(correct >= total):
                pass
        for x in c2:
            wx = 0
            i = 0
            j = 0
            for xi in x:
                if(i < len(w)):
                    wx += xi * w[i]
                    i += 1
            if (wx > 0):
                for wj in w:
                    if(j < len(w)):
                        w[j] -= step * x[j]
                        j+=1
                correct = 1
            else:
                correct += 1
            if(correct >= total):
                pass
    print("Iterations: {0}".format(iteration))
    print("=======Final Weights========================")
    i = 0
    for wi in w:
        print("w{0}: {1}".format(i,str(wi)))
        i+=1
    print("=======Final Dot Products=============")
    for x in c1:
        wx = 0
        i = 0
        for xi in x:
            if(i < len(w)):
                wx += xi * w[i]
                i += 1
        print(wx)
    print("======================================")
    for x in c2:
        wx = 0
        i = 0
        for xi in x:
            if(i < len(w)):
                wx += xi * w[i]
                i += 1
        print(wx)
    print("=======End=================================")

print("=======Dataset 1==============================================")
c1 = [[1, 1, 3, 5], [1, 2, 3, 10], [1, 3, 5, 9]]
c2 = [[1, -2, -1,-7], [1, -3, -3,-5], [1, -4, 4,-10]]
linear_classifier(c1,c2,0.6,1000)
print("=======Dataset 2(not linear separable)==============================================")
c1 = [[1, 1, 3, 5], [1, -3, -3,-5], [1, 3, 5, 9]]
c2 = [[1, -2, -1,-7], [1, 2, 3, 10], [1, -4, 4,-10]]
linear_classifier(c1,c2,0.3,1000000)