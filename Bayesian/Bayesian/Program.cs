using System.Linq;

List<Dictionary<string, string>> dataSet = new List<Dictionary<string, string>>()
{
    new Dictionary<string,string>(){ {"Assignment", "Good"}, {"Project", "A"}, {"Exam", "High"}, {"Label", "Pass"}},
    new Dictionary<string,string>(){ {"Assignment", "Good"}, {"Project", "B"}, {"Exam", "High"}, {"Label", "Pass"}},
    new Dictionary<string,string>(){ {"Assignment", "Bad"}, {"Project", "B"}, {"Exam", "Low"}, {"Label", "Fail"}},
    new Dictionary<string,string>(){ {"Assignment", "Bad"}, {"Project", "C"}, {"Exam", "High"}, {"Label", "Fail"}},
    new Dictionary<string,string>(){ {"Assignment", "Good"}, {"Project", "C"}, {"Exam", "Low"}, {"Label", "Fail"}},
    new Dictionary<string,string>(){ {"Assignment", "Good"}, {"Project", "C"}, {"Exam", "High"}, {"Label", "Pass"}},
    new Dictionary<string,string>(){ {"Assignment", "Bad"}, {"Project", "B"}, {"Exam", "High"}, {"Label", "Pass"}},
    new Dictionary<string,string>(){ {"Assignment", "Good"}, {"Project", "A"}, {"Exam", "Low"}, {"Label", "Pass"}},
    new Dictionary<string,string>(){ {"Assignment", "Bad"}, {"Project", "A"}, {"Exam", "Low"}, {"Label", "Fail"}},
    new Dictionary<string,string>(){ {"Assignment", "Good"}, { "Project", "B" }, { "Exam", "Low" }, {"Label", "Pass"}}
};

double Prior(string c,string ci)
{
    int total = dataSet.Count;
    double count = 0.0f;
    foreach(Dictionary<string, string> student in dataSet)
    {
        if(student.TryGetValue(c, out string? result))
        {
            if (result == ci) count++;
        }
    }
    return count / total;
}

double Likelihood(string f, string fi, string c, string ci)
{
    double cCount = 0.0f;
    double fCount = 0.0f;
    foreach (Dictionary<string, string> student in dataSet)
    {
        if (student.TryGetValue(c, out string? result))
        {
            if (result == ci)
            {
                if (student.TryGetValue(f, out string? result2))
                {
                    if (result2 == fi) fCount++;   
                }
                cCount++;
            }
        }
    }
    if (cCount > 0) return (double)fCount / cCount;
    return 0;
}

double Posterior(string c, string ci, Dictionary<string, string> featureDictonary)
{
    return featureDictonary.Keys.Aggregate(Prior(c, ci),
        (accmulate, nextKey) => accmulate *= Likelihood(nextKey, featureDictonary[nextKey], c, ci));
}

Dictionary<string, string> student = new Dictionary<string, string>()
{
    {"Assignment", "Bad"}, {"Project", "A"}, {"Exam", "High"} 
};
Console.WriteLine(Posterior("Label", "Pass", student));
Console.WriteLine(Posterior("Label", "Fail", student));