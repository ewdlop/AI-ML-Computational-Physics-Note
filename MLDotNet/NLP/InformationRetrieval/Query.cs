namespace InformationRetrieval;

public static class Query
{
    public static HashSet<int>? PhraseQuery(this Dictionary<string, (int DocumentFrequnecy, HashSet<int> DocumentIDs)> invertedIndex, string term) => 
        invertedIndex.TryGetValue(term, out (int DocumentFrequnecy, HashSet<int> DocumentIDs) documents)
            ? documents.DocumentIDs
            : null;

    public static Dictionary<int, List<int>>? PhraseQuery(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex, string term) =>
        positionalIndex.TryGetValue(term, out (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions) documents)
            ? documents.DocumentTermPositions
            : null;
    
    public static IEnumerable<HashSet<int>> FuzzyPhraseQuery(this Dictionary<string, (int DocumentFrequnecy, HashSet<int> DocumentIDs)> invertedIndex, string term, int threshold) =>
        invertedIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceWithinThreshold(term, threshold))
            .Select(s => s.Value.DocumentIDs);

    public static IEnumerable<Dictionary<int, List<int>>> FuzzyPhraseQuery(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex, string term, int threshold) =>
        positionalIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceWithinThreshold(term, threshold))
        .Select(s => s.Value.DocumentTermPositions);
    
    public static IEnumerable<string> FuzzyPhraseQueryMataches(this Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex, string term, int threshold) =>
        positionalIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceWithinThreshold(term, threshold))
            .Select(s => s.Key);

}