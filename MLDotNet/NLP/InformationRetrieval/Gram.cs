namespace InformationRetrieval;

public static class Gram
{
    //2-gram
    public static string[] TwoGram(this string[] words)
    {
        string[] twoGram = new string[words.Length - 1];
        for (int i = 0; i < words.Length - 1; i++)
        {
            twoGram[i] = $"{words[i]} {words[i + 1]}";
        }
        return twoGram;
    }

    //2-gram
    public static string[] TwoGram(this string sentence)
    {
        var Separators = new char[] { ' ', '\t', '\n', '\r', '\v', '\f' };
        string[] words = sentence.Split(Separators, StringSplitOptions.RemoveEmptyEntries);
        return TwoGram(words);
    }
}