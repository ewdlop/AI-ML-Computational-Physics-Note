namespace NLPRedo;

static class StringExtension
{
    public static IEnumerable<string> SplitAndKeep(
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
    public static IEnumerable<ReadOnlyMemory<char>> SplitAndKeepSpan(
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

}