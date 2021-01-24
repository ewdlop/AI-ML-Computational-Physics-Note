import random
import math
import sys

def distance(x1, x2):
    return ((abs(x1[0] - x2[0]))**2 + (abs(x1[1] - x2[1])**2)) **(1/2)

data_points = [(1,7),(1,11),(3,17),(7,18),(8,4),(8,12),(11,7),(12,14),(13,17),(16,11)]

#run all 1,2,3..,10-means clustering
for K in range(1,len(data_points)+1):
    print("==============================")
    print("K: {0}".format(K))

    # pick random k pointS
    C = random.sample(data_points,K)

    #prevent infinite loop criteria
    iteration = 1000000

    #converage criteria for minmumal decrease in sum of square errors
    min_decrease_sse = 1e-5

    #intital conditions for comparsions
    previous_sse = float("inf")
    current_iteration = 0
    
    while(True):
        current_iteration +=1
        current_sse = 0.0
        
        #create cluster membership dictonary for each centroids
        cms_dict = {}
        for c in C:
            cms_dict[c] = []

        for x in data_points:
            max = float("inf")
            closest = ()
            #compute the distance from x to each centroid
            for c in C:
                d= distance(x,c)
                if(d <= max):
                    max = d
                    closest = c

            #assign x to the closet centroid and its cluster memberships
            cms_dict[closest].append(x)
        
        #recomputing new centroids
        C = []
        for cm in cms_dict:
            cm_total_distance = 0.0
            new_c_x = 0.0
            new_c_y = 0.0
            
            #recompute the centroids using the current cluster memberships
            for x in cms_dict[cm]:
                new_c_x += x[0]
                new_c_y += x[1]
            new_c_x /= len(cms_dict[cm])
            new_c_y /= len(cms_dict[cm])
            C.append((new_c_x,new_c_y))
            
            #calucation the sum of squared error
            for x in cms_dict[cm]:
                cm_total_distance += distance(x,(new_c_x,new_c_y))**2
            current_sse += cm_total_distance

        #getting the decrease value in the sse
        if(previous_sse - current_sse <= min_decrease_sse or current_iteration > iteration):
            print("Final SSE: {0}".format(current_sse))
            print("Final Iteration: {0}".format(current_iteration))
            i = 0
            for cm in cms_dict:
                i+=1
                print("Cluster {0}: {1}".format(i,cms_dict[cm]))
            break
        else:
            print("Current SSE: {0}".format(current_sse))
            previous_sse = current_sse