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

    public static IEnumerable<(int, double)> TFIDFScore2(this Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions)> positionalIndex, string[] terms, int totalDocumentCount) =>
    terms.SelectMany(t => positionalIndex.TFIDFs(t,totalDocumentCount))
        .GroupBy(f => f.DocumentID)
        .Select(s => (s.Key,s.Sum(t => t.TFIDF)));
}