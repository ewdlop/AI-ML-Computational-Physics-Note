using System.Collections.Generic;

namespace InformationRetrieval;

public static class Indexer
{
    private static readonly char[] Separators = new char[] { ' ', '\t', '\n' };
    //private readonly Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> termDocumentPositions)> positionalIndex = new();


    /// <summary>
    /// Inverted Index
    /// </summary>
    /// <param name="documents"></param>
    public static Dictionary<string, (int documentFrequnecy, HashSet<int> documentIDs)> InvertedIndex(this string[] documents)
    {
        Dictionary<string, (int documentFrequnecy, HashSet<int> documentIDs)> invertedIndex = new();
        for (int i = 0; i < documents.Length; i++)
        {
            string[] words = documents[i].Split(Separators, StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < words.Length; j++)
            {
                string word = words[j];
                if (invertedIndex.TryGetValue(word, out (int documentFrequnecy, HashSet<int> documentIDs) existing))
                {
                    invertedIndex[word] = (existing.documentFrequnecy + 1,
                        existing.documentIDs.Append(i).OrderBy(i => i).ToHashSet());
                }
                else
                {
                    invertedIndex[word] = (1, new HashSet<int>(i));
                }
            }
        }
        return invertedIndex;
    }

    /// <summary>
    /// Positional Index
    /// https://www.youtube.com/watch?v=QVVvx_Csd2I&list=PLaZQkZp6WhWwoDuD6pQCmgVyDbUWl_ZUi&index=6
    /// </summary>
    /// <param name="documents"></param>
    public static Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions)> PositionalIndex(this string[] documents)
    {
        Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions)> positionalIndex = new();
        for (int i = 0; i < documents.Length; i++)
        {
            string[] words = documents[i].Split(Separators, StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < words.Length; j++)
            {
                string word = words[j];
                if (positionalIndex.TryGetValue(word, out (int documentFrequnecy, Dictionary<int, List<int>> documentTermPositions) existing))
                {
                    if (existing.documentTermPositions.TryGetValue(i, out List<int>? existingPositions))
                    {
                        existingPositions.Add(j);
                    }
                    else
                    {
                        existing.documentTermPositions[i] = new List<int>(1) { j };
                    }
                }
                else
                {
                    positionalIndex[word] = (1, new Dictionary<int, List<int>>
                    {
                        { i, new List<int>(1) { j } }
                    });
                }
            }
        }
        return positionalIndex;
    }

    public static IEnumerable<HashSet<int>> PhraseQuery(this Dictionary<string, (int documentFrequnecy, HashSet<int> documentIDs)> invertedIndex, string search, int threshold)
    {
        return invertedIndex.Where(s => s.Key.OptimalDamerauLevenshteinDistanceThreshold(search, threshold))
            .Select(s => s.Value.documentIDs);
    }

    
}