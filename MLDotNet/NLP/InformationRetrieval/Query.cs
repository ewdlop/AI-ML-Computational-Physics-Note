namespace InformationRetrieval;

public static class Query
{
    public static IEnumerable<HashSet<int>> FuzzyPhraseQuery(this Dictionary<string, (int DocumentFrequnecy, HashSet<int> DocumentIDs)> invertedIndex, string term, int threshold) =>
        invertedIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceThreshold(term, threshold))
            .Select(s => s.Value.DocumentIDs);

    public static HashSet<int>? PhraseQuery(this Dictionary<string, (int DocumentFrequnecy, HashSet<int> DocumentIDs)> invertedIndex, string term) => 
        invertedIndex.TryGetValue(term, out (int DocumentFrequnecy, HashSet<int> DocumentIDs) documents)
            ? documents.DocumentIDs
            : null;
}