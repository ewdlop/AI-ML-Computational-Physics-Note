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
    public static Dictionary<string, (int DocumentFrequnecy, HashSet<int> DocumentIDs)> InvertedIndex(this string[] documents)
    {
        Dictionary<string, (int documentFrequnecy, HashSet<int> documentIDs)> invertedIndex = new();
        for (int i = 0; i < documents.Length; i++)
        {
            string[] words = documents[i].Split(Separators, StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < words.Length; j++)
            {
                string word = words[j];
                if (invertedIndex.TryGetValue(word, out (int DocumentFrequnecy, HashSet<int> DocumentIDs) existing))
                {
                    invertedIndex[word] = (existing.DocumentFrequnecy + 1,
                        existing.DocumentIDs.Append(i).OrderBy(i => i).ToHashSet());
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
    public static Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> PositionalIndex(this string[] documents)
    {
        Dictionary<string, (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions)> positionalIndex = new();
        for (int i = 0; i < documents.Length; i++)
        {
            string[] words = documents[i].Split(Separators, StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < words.Length; j++)
            {
                string word = words[j];
                if (positionalIndex.TryGetValue(word, out (int DocumentFrequnecy, Dictionary<int, List<int>> DocumentTermPositions) existing))
                {
                    if (existing.DocumentTermPositions.TryGetValue(i, out List<int>? existingPositions))
                    {
                        existingPositions.Add(j);
                    }
                    else
                    {
                        existing.DocumentTermPositions[i] = new List<int>(1) { j };
                        positionalIndex[word] = (existing.DocumentFrequnecy + 1, existing.DocumentTermPositions);
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
}   
