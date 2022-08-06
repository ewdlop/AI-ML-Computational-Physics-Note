public class Test
{
    //Inverted Index
    public static Dictionary<string, List<int>> InvertedIndex(string[] words)
    {
        Dictionary<string, List<int>> result = new Dictionary<string, List<int>>();
        for (int i = 0; i < words.Length; i++)
        {
            if (result.ContainsKey(words[i]))
            {
                result[words[i]].Add(i);
            }
            else
            {
                result.Add(words[i], new List<int>(1) { i });
            }
        }
        return result;
    }
}
