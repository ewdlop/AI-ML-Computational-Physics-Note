namespace NLP.Extensions;

public static class StringArrayExtension
{
    //Inverted Index
    public static Dictionary<string, List<int>> InvertedIndex(this string[] array)
    {
        Dictionary<string, List<int>> dict = new Dictionary<string, List<int>>();
        for (int i = 0; i < array.Length; i++)
        {
            var words = array[i].Split(' ');
            foreach (var word in words)
            {
                if (!dict.ContainsKey(word))
                {
                    dict.Add(word, new List<int>());
                }
                dict[word].Add(i);
            }
        }
        return dict;
    }

    //Inverted Index with location
    public static Dictionary<string, List<(int, int)>> InvertedIndexWithLocation(this string[] array)
    {
        Dictionary<string, List<(int, int)>> dict = new Dictionary<string, List<(int, int)>>();
        for (int i = 0; i < array.Length; i++)
        {
            var words = array[i].Split(' ');
            for (int i1 = 0; i1 < words.Length; i1++)
            {
                string? word = words[i1];
                if(dict.TryGetValue(word, out List<(int,int)>? value))
                {
                    value.Add((i, i1));
                }
                else
                {
                    dict.Add(word, new List<(int, int)>() { (i, i1) });
                }
            }
        }
        return dict;
    }

    //Inverted Index with location, lexicographic order
    public static Dictionary<string, List<(int, int)>> InvertedIndexWithLocationLexicographicOrder(this string[] array)
    {
        Dictionary<string, List<(int, int)>> dict = new Dictionary<string, List<(int, int)>>();
        for (int i = 0; i < array.Length; i++)
        {
            var words = array[i].Split(' ');
            for (int i1 = 0; i1 < words.Length; i1++)
            {
                string? word = words[i1];
                if (dict.TryGetValue(word, out List<(int, int)>? value))
                {
                    value.Add((i, i1));
                }
                else
                {
                    dict.Add(word, new List<(int, int)>() { (i, i1) });
                }
            }
        }
        return dict.OrderBy(x => x.Key).ToDictionary(x => x.Key, x => x.Value);
    }

    //Levenshtein distance search with location with custom threshold
    public static Dictionary<string, List<(int, int)>> LevenshteinDistanceInvertedIndexWithLocation(this string[] array, string search, int threshold)
    {
        Dictionary<string, List<(int, int)>> dict = new Dictionary<string, List<(int, int)>>();
        for (int i = 0; i < array.Length; i++)
        {
            string[]? words = array[i].Split(' ');
            for (int i1 = 0; i1 < words.Length; i1++)
            {
                string word = words[i1];
                if (word.OptimalDamerauLevenshteinDistance(search) <= threshold)
                {
                    if (dict.TryGetValue(word, out List<(int, int)>? value))
                    {
                        value.Add((i, i1));
                    }
                    else
                    {
                        dict.Add(word, new List<(int, int)>() { (i, i1) });
                    }
                }
            }
        }
        return dict;
    }
}