using System.Collections.Concurrent;
using System.Collections.Immutable;

namespace InformationRetrieval.IR;

public static class IR
{
    private static readonly char[] Separators = new char[] { ' ', '\t', '\n' };

    /// <summary>
    /// Inverted Index
    /// </summary>
    /// <param name="documents"></param>
    public static Dictionary<string, (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs)> InvertedIndex(this string[] documents)
    {
        Dictionary<string, (int documentFrequnecy, ImmutableSortedSet<int> documentIDs)> invertedIndex = new();
        for (int i = 0; i < documents.Length; i++)
        {
            string[] words = documents[i].Split(Separators, StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < words.Length; j++)
            {
                string word = words[j];
                if (invertedIndex.TryGetValue(word, out (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs) existing))
                {
                    invertedIndex[word] = (existing.DocumentFrequnecy + 1, existing.DocumentIDs.Add(i));
                }
                else
                {
                    invertedIndex[word] = (1, ImmutableSortedSet.Create(i));
                }
            }
        }
        return invertedIndex;
    }

    public static int? First(this Dictionary<string, (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs)> invertedIndex, string term)
    {
        if (invertedIndex.TryGetValue(term, out (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs) documents))
        {
            if (documents.DocumentIDs.Count > 0)
            {
                return documents.DocumentIDs[0];
            }
            else
            {
                return null;
            }
        }
        else
        {
            return null;
        }
    }

    public static int? Last(this Dictionary<string, (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs)> invertedIndex, string term)
    {
        if (invertedIndex.TryGetValue(term, out (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs) documents))
        {
            if (documents.DocumentIDs.Count > 0)
            {
                return documents.DocumentIDs[^1];
            }
            else
            {
                return null;
            }
        }
        else
        {
            return null;
        }
    }

    public static int? Next(this Dictionary<string, (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs)> invertedIndex, string term)
    {
        if (invertedIndex.TryGetValue(term, out (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs) documents))
        {
            return Next(documents.DocumentIDs, term);
        }
        else
        {
            return null;
        }
    }

    /// <summary>
    /// Exponentail or Galloping Search
    /// </summary>
    /// <param name="DocumentIDs"></param>
    /// <param name="term"></param>
    /// <param name="current"></param>
    /// <returns></returns>
    private static int Next(ImmutableSortedSet<int> DocumentIDs, string term, int current = 0)
    {
        int length = DocumentIDs.Count;
        if (length == 0 || DocumentIDs[^1] <= current)
        {
            return int.MaxValue;
        }
        ConcurrentDictionary<string, int> cache = new();
        if (DocumentIDs[0] > current)
        {
            cache.AddOrUpdate(term, 0, 
                (key, value) => 0);
            return DocumentIDs[0];
        }
        int low;
        if (cache.TryGetValue(term, out int index) && index > 0 && DocumentIDs[index] <= current)
        {
            low = index;
        }
        else
        {
            low = 0;
        }
        int jump = 1;
        int high = low + jump;
        while (high < length && DocumentIDs[high] <= current)
        {
            low = high;
            jump = jump == 0 ? 1 : jump * 2;
            high = low + jump;
        }
        if (high > length)
        {
            high = length;
        }
        cache.AddOrUpdate(term, DocumentIDs.BinarySearch(low, high, current), 
            (key, value) => DocumentIDs.BinarySearch(low, high, current));
        return 0;
    }
    public static int? Previous(this Dictionary<string, (int DocumentFrequnecy, ImmutableSortedSet<int> DocumentIDs)> invertedIndex, string term)
    {
        throw new NotImplementedException();
    }
    

    private static int BinarySearch(this ImmutableSortedSet<int> DocumentIDs, int low,
        int high, int current)
    {        
        while (high - low > 1)
        {
            int mid = (low + high) / 2;
            if (DocumentIDs[mid] <= current)
            {
                low = mid;
            }
            else
            {
                high = mid;
            }
        }
        return high;
    }
}