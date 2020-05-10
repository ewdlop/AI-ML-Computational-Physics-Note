#p(Pass|Bad, A, High) = p(Pass) * p(Bad|Pass) * p(A|Pass) * p(High|Pass)
#p(Fail|Bad, A, High) = p(Fail) * p(Bad|Fail) * p(A|Fail) * p(High|Fail)

dataset = [
    {"Assignment": "Good", "Project": "A", "Exam": "High", "Label": "Pass"},
    {"Assignment": "Good", "Project": "B", "Exam": "High", "Label": "Pass"},
    {"Assignment": "Bad", "Project": "B", "Exam": "Low", "Label": "Fail"},
    {"Assignment": "Bad", "Project": "C", "Exam": "High", "Label": "Fail"},
    {"Assignment": "Good", "Project": "C", "Exam": "Low", "Label": "Fail"},
    {"Assignment": "Good", "Project": "C", "Exam": "High", "Label": "Pass"},
    {"Assignment": "Bad", "Project": "B", "Exam": "High", "Label": "Pass"},
    {"Assignment": "Good", "Project": "A", "Exam": "Low", "Label": "Pass"},
    {"Assignment": "Bad", "Project": "A", "Exam": "Low", "Label": "Fail"},
    {"Assignment": "Good", "Project": "B", "Exam": "Low", "Label": "Pass"}
]

#P(c=ci)
def prior(c,ci):
    total = len(dataset)
    count = 0.0
    for student in dataset:
        if(student[c] is not None and student[c] == ci):
            count+=1
    return count/total

#P(f=fi|c=ci)
def likelihood(f, fi, c, ci):
    c_count = 0.0
    f_count = 0.0
    for student in dataset:
        if(student[c] is not None and student[c] == ci):
            if(student[f] is not None and student[f] == fi):
                f_count+=1 
            c_count+=1
    if(c_count > 0.0):
        return f_count/c_count
    return None
        
#P(C=ci|f1,f2,f3,...fn) = p(ci) * p(f1|ci) * p(f2|ci) * ... * p(fn|ci) 
def posterior(c,ci, feature_dictonary):
    p_c = prior(c, ci)
    for key in feature_dictonary.keys():
        p_c *= likelihood(key,feature_dictonary[key],c,ci)
    return p_c

print("Probability for passing:{0}".format(posterior("Label","Pass",{"Assignment": "Bad", "Project": "A", "Exam": "High"})))
print("Probability for failing:{0}".format(posterior("Label","Fail",{"Assignment": "Bad", "Project": "A", "Exam": "High"})))