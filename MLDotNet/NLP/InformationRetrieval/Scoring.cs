namespace InformationRetrieval;

public static class Scoring
{
    public static IEnumerable<double> Score(this Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions)> positionalIndex, string[] terms) => 
        terms.SelectMany(t => positionalIndex.LogTermFrequencies(t))
            .GroupBy(f => f.DocumentID)
            .Select(s => s.Sum(t => t.LogTermFrequency));

    public static IEnumerable<double> TFIDFScore(this Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions)> positionalIndex, string[] terms, int totalDocumentCount) =>
    terms.SelectMany(t => positionalIndex.TFIDFs(t, totalDocumentCount))
        .GroupBy(f => f.DocumentID)
        .Select(s => s.Sum(t => t.TFIDF));

    public static IEnumerable<(int DocumentID, double score)> TFIDFScore2(this Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions)> positionalIndex, string[] terms, int totalDocumentCount) =>
        terms.SelectMany(t => positionalIndex.TFIDFs(t,totalDocumentCount))
            .GroupBy(f => f.DocumentID)
            .Select(s => (s.Key,s.Sum(t => t.TFIDF)));

    /// <summary>
    /// Vector space mode, not done
    /// </summary>
    /// <param name="terms"></param>
    /// <param name="documents"></param>
    /// <returns></returns>
    public static IEnumerable<double> CosineSimilarityScore(string[] terms, string[] documents)
    {
        Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> termsIndex = terms.PositionalIndex();
        Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> documentsIndex = documents.PositionalIndex();
        IEnumerable<(int Key, IEnumerable<double>)> normalizeQ = terms.SelectMany(t => termsIndex.TFIDFs(t, documents.Length))
            .GroupBy(f => f.DocumentID).Select(s => (s.Key, s.Select(d => d.TFIDF / Math.Sqrt(s.Sum(t => t.TFIDF)))));
        IEnumerable<(int Key, IEnumerable<double>)> normalizeD = terms.SelectMany(t => documentsIndex.LogTermFrequencies(t))
            .GroupBy(f => f.DocumentID).Select(s => (s.Key, s.Select(d => d.LogTermFrequency / Math.Sqrt(s.Sum(t => t.LogTermFrequency)))));
        //normalizeQ.Join(normalizeD, s => s.Key, d => d.Key, (s, d) => (s.Key, s.Item2.Zip(d.Item2, (q, d) => q * d))).Select(s => Math.Sqrt(s.Item2.Sum(t => t * t)));
        return null;
    }
}