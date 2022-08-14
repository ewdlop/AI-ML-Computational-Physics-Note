namespace InformationRetrieval;

public static class Frequency
{
    /// <summary>
    /// Jaccard Coefficent
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static double JaccardCoefficent(this string s, string t)
    {
        return s.Intersect(t).Count() / (double)s.Union(t).Count();
    }

    /// <summary>
    /// Jaccard Coefficent
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static double JaccardCoefficent(this string[] s, string[] t)
    {
        return s.Intersect(t).Count() / (double)s.Union(t).Count();
    }

    /// <summary>
    /// Jaccard Coefficent
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static double JaccardCoefficent(this HashSet<string> s, HashSet<string> t)
    {
        return s.Intersect(t).Count() / (double)s.Union(t).Count();
    }
    
    public static IEnumerable<(int, int)> FuzzyTermFrequencies(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> PositionalIndex, string term, int threshold)
    {
        return PositionalIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceWithinThreshold(term, threshold))
            .SelectMany(s => s.Value.DocumentTermPositions).Select(s => (s.Key, s.Value.Count));
    }

    public static IEnumerable<(int DocumentID, int TermFrequencies)> TermFrequencies(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> PositionalIndex, string term)
    {
        if (PositionalIndex.TryGetValue(term, out (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions) Documents))
        {
            return Documents.DocumentTermPositions.Select(s => (s.Key, s.Value.Count));
        }
        else
        {
            return Enumerable.Empty<(int DocumentID, int TermFrequencies)>();
        }
    }

    public static IEnumerable<(int, double)> FuzzyLogTermFrequencies(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> PositionalIndex, string term, int threshold)
    {
        return PositionalIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceWithinThreshold(term, threshold))
            .SelectMany(s => s.Value.DocumentTermPositions).Select(s => (s.Key, 1 + Math.Log10(s.Value.Count)));
    }

    public static IEnumerable<(int DocumentID, double LogTermFrequency)> LogTermFrequencies(this Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex, string term)
    {
        if (positionalIndex.TryGetValue(term, out (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions) documents))
        {
            return documents.DocumentTermPositions.Select(s => (s.Key, 1 + Math.Log10(s.Value.Count)));
        }
        else
        {
            return Enumerable.Empty<(int DocumentID, double Score)>();
        }
    }

    /// <summary>
    /// Allowing less common terms having higher score
    /// </summary>
    /// <param name="positionalIndex"></param>
    /// <param name="term"></param>
    /// <param name="totalDocumentCount"></param>
    /// <returns></returns>
    public static double InverseDocumentFrequency(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex, string term, int totalDocumentCount)
    {
        if (positionalIndex.TryGetValue(term, out (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions) documents))
        {
            return Math.Log10(totalDocumentCount / documents.DocumentFrequnecy);
        }
        else
        {
            return 0;
        }
    }

    //TermFrequenciesInverseDocumentFrequency
    public static IEnumerable<(int DocumentID, double TFIDF)> TFIDFs(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex, string term, int totalDocumentCount)
    {
        if (positionalIndex.TryGetValue(term, out (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions) documents))
        {
            return documents.DocumentTermPositions.Select(s => (s.Key, (1 + Math.Log10(s.Value.Count)) * Math.Log10((double)totalDocumentCount / documents.DocumentFrequnecy)));
        }
        else
        {
            return Enumerable.Empty<(int DocumentID, double Score)>();
        }
    }
}
