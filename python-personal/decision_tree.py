from math import log2

def entropy(p,n):
    if p == 0 or n == 0:
        return 0
    else:
        return -1 * p/(p+n) * log2(p/(p+n)) - n/(p+n) *log2(n/(p+n))

def info_gain(hy,list_postive, list_negative):
    p1 = 0
    n1 = 0
    p2 = 0
    n2 = 0
    for i in range(len(list_postive)):
        if(i == 1):
            p1 = p1 + 1
        if(i == 0):
            n1 = n1 + 1
    for i in range(len(list_negative)):
        if(i == 1):
            p2 = p1 + 1
        if(i == 0):
            n2 = n1 + 1
    return hy - (len(list_postive)/(len(list_postive) + len(list_negative)) * entropy(p1, n1) + len(list_negative)/(len(list_postive) + len(list_negative)) * entropy(p2,n2))