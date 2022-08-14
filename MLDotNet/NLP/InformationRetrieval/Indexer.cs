namespace InformationRetrieval;

public class Indexer
{
    private static readonly char[] Separators = new char[] { ' ', '\t', '\n' };
    //private readonly Dictionary<string, (int documentFrequnecy, Dictionary<int, List<int>> termDocumentPositions)> positionalIndex = new();

    /// <summary>
    /// Positional Index
    /// https://www.youtube.com/watch?v=QVVvx_Csd2I&list=PLaZQkZp6WhWwoDuD6pQCmgVyDbUWl_ZUi&index=6
    /// </summary>
    /// <param name="documents"></param>
    public static void PositionalIndex(string[] documents)
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
    }
}