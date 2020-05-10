import random
import math
import sys
import statistics

def mahattan_distance_one_dimension(x1,x2):
    return abs(x1-x2)

def minkowski_distance(x1, x2, power):
    return ((abs(x1[0] - x2[0]))**power + (abs(x1[1] - x2[1])**power)) **(1.0/power)

def chebyshev_distance(x1, x2, power=0):
    return max(abs(x1[0]-x2[0]),abs(x1[1]-x2[1]))

one_dimensional_data_points = [5,13,4,6,15,13,32,14,6,10,12,31,12,41,13]
data_points = [(1,7),(1,-11),(3,17),(7,18),(-8,4),(8,12),(11,-7),(12,14),(13,71),(-16,11),(13,1),(-9,2),(-5,3),(0,12)]

def k_means_clustering(K,func,power=0):
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
        
        #create cluster membership dictonary for each centroid
        cms_dict = {}
        for c in C:
            cms_dict[c] = []

        for x in data_points:
            max_dist = float("inf")
            closest = ()
            #compute the distance from x to each centroid
            for c in C:
                d= func(x,c,power)
                if(d <= max_dist):
                    max_dist = d
                    closest = c

            #assign x to the closet centroid and its cluster memberships
            cms_dict[closest].append(x)
        
        #recomputing new centroids
        C = []
        for cm in cms_dict:
            cm_total_distance = 0.0
            new_m_x = 0.0
            new_m_y = 0.0
            
            #recompute the centroids using the current cluster memberships
            for x in cms_dict[cm]:
                new_m_x += x[0]
                new_m_y += x[1]
            new_m_x /= len(cms_dict[cm])
            new_m_y /= len(cms_dict[cm])
            C.append((new_m_x,new_m_y))
            
            #calucation the sum of squared error
            for x in cms_dict[cm]:
                cm_total_distance += func(x,(new_m_x,new_m_y),power)**2
            current_sse += cm_total_distance

        #getting the decrease value in the sse
        if(previous_sse - current_sse <= min_decrease_sse or current_iteration > iteration):
            print("Final SSE: {0}".format(current_sse))
            print("Final Iteration: {0}".format(current_iteration))
            i = 0
            silhouetee_coefficent = 0.0
            #calculate average silhouetee coefficent
            for cm in cms_dict:
                i+=1
                abs = []
                if(len(cms_dict[cm]) != 1):
                    if(K > 1):
                        m = 0
                        for xi in cms_dict[cm]:
                            total_distance = 0
                            for xj in cms_dict[cm]:
                                if(xi != xj):
                                    total_distance += func(xi,xj,power)  
                            ai = total_distance//(len(cms_dict[cm])-1)
                            bi = None
                            for cm2 in cms_dict:
                                if( cm !=cm2 ):
                                    total_distance = 0
                                    for xj in cms_dict[cm2]:
                                        total_distance += func(xi,xj,power)
                                    average = total_distance//len(cms_dict[cm2])
                                    if(bi is None):
                                        bi = average
                                    else:
                                        if(average < bi ):
                                            bi = average
                            si = float(bi - ai) / max(ai,bi)
                            silhouetee_coefficent += si
                            dict ={}
                            dict["a{0}".format(m)] = ai
                            dict["b{0}".format(m)] = bi
                            dict["s{0}".format(m)] = si
                            abs.append(dict)
                            m+=1
                else:
                    dict ={}
                    dict["a0"] = "Undefined"
                    dict["b0"] = "Undefined"
                    dict["s0"] = "0 by defintion"
                    abs.append(dict)
                print("Cluster {0}: {1}".format(i,cms_dict[cm]))
                print(abs)
                print("------------------------------------------------")
            if( K > 1):
                print("Average Silhouetee Coefficent:{0}".format(silhouetee_coefficent/len(data_points)))
            else:
                print("Silhouetee Coefficent is not defined for K = 1")
            break
        else:
            print("Current SSE: {0}".format(current_sse))
            previous_sse = current_sse
        

def k_median_clustering(K):
    print("==============================")
    print("K: {0}".format(K))

    # pick random k pointS
    C = random.sample(one_dimensional_data_points,K)

    #prevent infinite loop criteria
    iteration = 1000000

    #converage criteria for minmumal decrease in sum of errors
    min_decrease_se = 1e-5

    #intital condiions for comparsions
    previous_se = float("inf")
    current_iteration = 0
    
    while(True):
        current_iteration +=1
        current_se = 0.0
        
        #create cluster membership dictonary for each median
        cms_dict = {}
        for c in C:
            cms_dict[c] = []

        for x in one_dimensional_data_points:
            max_dist = float("inf")
            closest = None
            #compute the distance from x to each median
            for c in C:
                d = mahattan_distance_one_dimension(x,c,)
                if(d <= max_dist):
                    max_dist = d
                    closest = c

            #assign x to the closet median and its cluster memberships
            cms_dict[closest].append(x)
        
        #recomputing new centroids
        C = []
        for cm in cms_dict:
            cm_total_distance = 0.0
            new_m_x = 0.0
            
            #recompute the median using the current cluster memberships
            new_m_x = statistics.median(cms_dict[cm])
            C.append(new_m_x)
            
            #calucation the sum of error
            for x in cms_dict[cm]:
                cm_total_distance += mahattan_distance_one_dimension(x,(new_m_x))
            current_se += cm_total_distance

        #getting the decrease value in the sse
        if(previous_se - current_se <= min_decrease_se or current_iteration > iteration):
            print("Final SSE: {0}".format(current_se))
            print("Final Iteration: {0}".format(current_iteration))
            i = 0
            silhouetee_coefficent = 0.0
            #calculate average silhouetee coefficent
            for cm in cms_dict:
                i+=1
                abs = []
                if(len(cms_dict[cm]) != 1):
                    if(K > 1):
                        m = 0
                        for xi in cms_dict[cm]:
                            total_distance = 0
                            for xj in cms_dict[cm]:
                                if(xi != xj):
                                    total_distance += mahattan_distance_one_dimension(xi,xj)  
                            ai = total_distance//(len(cms_dict[cm])-1)
                            bi = None
                            for cm2 in cms_dict:
                                if( cm !=cm2 ):
                                    total_distance = 0
                                    for xj in cms_dict[cm2]:
                                        total_distance += mahattan_distance_one_dimension(xi,xj)
                                    average = total_distance//len(cms_dict[cm2])
                                    if(bi is None):
                                        bi = average
                                    else:
                                        if(average < bi ):
                                            bi = average
                            si = float(bi - ai) / max(ai,bi)
                            silhouetee_coefficent += si
                            dict ={}
                            dict["a{0}".format(m)] = ai
                            dict["b{0}".format(m)] = bi
                            dict["s{0}".format(m)] = si
                            abs.append(dict)
                            m+=1
                else:
                    dict ={}
                    dict["a0"] = "Undefined"
                    dict["b0"] = "Undefined"
                    dict["s0"] = "0 by defintion"
                    abs.append(dict)
                print("Cluster {0}: {1}, Median: {2}".format(i,cms_dict[cm],statistics.median(cms_dict[cm])))
                print(abs)
                print("------------------------------------------------")
            if( K > 1):
                print("Average Silhouetee Coefficent:{0}".format(silhouetee_coefficent / len(data_points)))
            else:
                print("Silhouetee Coefficent is not defined for K = 1")
            break
        else:
            print("Current SSE: {0}".format(current_se))
            previous_se = current_se

def k_medoids_clustering(K,func,power=0):
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
        
        #create cluster membership dictonary for each medoid
        cms_dict = {}
        for c in C:
            cms_dict[c] = []

        for x in data_points:
            max_dist = float("inf")
            closest = ()
            #compute the distance from x to each medoid
            for c in C:
                d= func(x,c,power)
                if(d <= max_dist):
                    max_dist = d
                    closest = c

            #assign x to the closet centroid and its cluster memberships
            cms_dict[closest].append(x)
        
        #recomputing new medoid
        C = []
        for m in cms_dict:
            max_dist = float("inf")
            new_c = None
            #recompute the medoids using the current cluster memberships
            for x1 in cms_dict[m]:
                cm_total_distance = 0.0
                for x2 in cms_dict[m]:
                    cm_total_distance += func(x1, x2, power)
                if(cm_total_distance <= max_dist):
                    max_dist = cm_total_distance
                    new_c = x1
            C.append(new_c)
            
            #calucation the sum of squared error
            cm_total_distance = 0.0
            for x in cms_dict[m]:
                cm_total_distance += func(x,new_c,power)**2
            current_sse += cm_total_distance

        #getting the decrease value in the sse
        if(previous_sse - current_sse <= min_decrease_sse or current_iteration > iteration):
            print("Final SSE: {0}".format(current_sse))
            print("Final Iteration: {0}".format(current_iteration))
            silhouetee_coefficent = 0.0
            i = 0
            #calculate average silhouetee coefficent
            for cm in cms_dict:
                i+=1
                abs = []
                if(len(cms_dict[cm]) != 1):
                    if(K > 1):
                        m = 0
                        for xi in cms_dict[cm]:
                            total_distance = 0
                            for xj in cms_dict[cm]:
                                if(xi != xj):
                                    total_distance += func(xi,xj,power)  
                            ai = total_distance//(len(cms_dict[cm])-1)
                            bi = None
                            for cm2 in cms_dict:
                                if( cm !=cm2 ):
                                    total_distance = 0
                                    for xj in cms_dict[cm2]:
                                        total_distance += func(xi,xj,power)
                                    average = total_distance//len(cms_dict[cm2])
                                    if(bi is None):
                                        bi = average
                                    else:
                                        if(average < bi ):
                                            bi = average
                            si = float(bi - ai) / max(ai,bi)
                            silhouetee_coefficent += si
                            dict ={}
                            dict["a{0}".format(m)] = ai
                            dict["b{0}".format(m)] = bi
                            dict["s{0}".format(m)] = si
                            abs.append(dict)
                            m+=1
                else:
                    dict ={}
                    dict["a0"] = "Undefined"
                    dict["b0"] = "Undefined"
                    dict["s0"] = "0 by defintion"
                    abs.append(dict)
                print("Cluster {0}: {1}".format(i,cms_dict[cm]))
                print(abs)
                print("------------------------------------------------")
            if( K > 1):
                print("Average Silhouetee Coefficent:{0}".format(silhouetee_coefficent/len(data_points)))
            else:
                print("Silhouetee Coefficent is not defined for K = 1")
            break
            
        else:
            print("Current SSE: {0}".format(current_sse))
            previous_sse = current_sse


def main(argv):
    algo = {
        "euclidean": lambda k,pow: k_means_clustering(k,minkowski_distance,2),
        "minkowski":  lambda k,pow: k_means_clustering(k,minkowski_distance,pow),
        "chebyshev": lambda k,pow,: k_means_clustering(k,chebyshev_distance),
        "median": lambda k,pow: k_median_clustering(k),
        "medoids": lambda k,pow: k_medoids_clustering(k,minkowski_distance,pow)
    }
    for k in range(1,int(argv[1])+1):
        algo[str(argv[0])](k,float(argv[1]) if (len(argv) > 2 and argv[2].replace('.','',1).isdigit()) else 2 )
        print("===========================================================================")

if __name__ == "__main__":
    if(len(sys.argv) == 1):
       print("Missing Arugments")
    else:
        main(sys.argv[1:])