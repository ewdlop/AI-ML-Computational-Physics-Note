namespace NLP.Extensions;

public static partial class StringExtension
{
    private static readonly Lazy<string[]> _delimiters = new(() => new string[] {" ", "   ", "\r\n" });
    public static IEnumerable<ReadOnlyMemory<char>> AsSplitAndNotKeepDelimter(
        this string inputString, params char[] delimiters)
    {
        int start = 0;
        while (true)
        {
            int nextDelimiter = inputString.IndexOfAny(delimiters, start);
            if (nextDelimiter < 0)
            {
                yield return inputString.AsMemory()[start..];
                break;
            }
            yield return inputString.AsMemory()[start..nextDelimiter];
            start = nextDelimiter + 1;
        }
    }

    public static IEnumerable<string> ToSplitAndNotKeepDelimter(
        this string inputString, params char[] delimiters)
    {
        int start = 0;
        while (true)
        {
            int nextDelimiter = inputString.IndexOfAny(delimiters, start);
            if (nextDelimiter < 0)
            {
                yield return inputString[start..];
                break;
            }
            yield return inputString[start..nextDelimiter];
            start = nextDelimiter + 1;
        }
    }

    public static IEnumerable<ReadOnlyMemory<char>> AsSplitAndKeepDelimter(
        this string inputString, params char[] delimiters)
    {
        int start = 0, index;

        while ((index = inputString.IndexOfAny(delimiters, start)) != -1)
        {
            if (index - start > 0)
                yield return inputString.AsMemory()[start..index];

            yield return inputString.AsMemory().Slice(index, 1);

            start = index + 1;
        }

        if (start < inputString.Length)
        {
            yield return inputString.AsMemory()[start..];
        }
    }

    public static IEnumerable<string> ToSplitAndKeepDelimter(
        this string inputString, params char[] delimiters)
    {
        int start = 0, index;

        while ((index = inputString.IndexOfAny(delimiters, start)) != -1)
        {
            if (index - start > 0)
                yield return inputString[start..index];

            yield return inputString.Substring(index, 1);

            start = index + 1;
        }

        if (start < inputString.Length)
        {
            yield return inputString[start..];
        }
    }

    public static IEnumerable<ReadOnlyMemory<char>> AsTokenizeSentence(this string text)
    {
        return text.Split(_delimiters.Value, StringSplitOptions.None)
            .SelectMany(o => o.AsSplitAndKeepDelimter(".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()));
    }
}