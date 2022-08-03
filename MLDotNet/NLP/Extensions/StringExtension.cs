namespace NLP.Extensions;

public static partial class StringExtension
{
    private const string DELIMITERS = ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'";
    private static readonly Lazy<string[]> _delimiters = new(() => new string[3] {" ", "   ", "\r\n" });
    public static IEnumerable<ReadOnlyMemory<char>> AsSplitMemoryAndNotKeepingDelimters(
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

    public static IEnumerable<string> ToSplitStringsNotKeepingDelimters(
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

    public static IEnumerable<ReadOnlyMemory<char>> AsSplitMemoryKeepingDelimter(
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

    public static IEnumerable<string> ToSplitStringKeepingDelimters(
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
            .SelectMany(o => o.AsSplitMemoryKeepingDelimter(DELIMITERS.ToArray()));
    }

    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    /// see also: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance
    /// default is case sensitive
    /// Levenshtein distance is a measure of the difference between two sequences.
    /// The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.
    /// Triangle inequality does not hold.
    /// Two Dimension matrix is used to calculate the distance.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int OptimalDamerauLevenshteinDistance(this string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[,] d = new int[n + 1, m + 1];

        // Step 1
        if (n == 0) return m;
        if (m == 0) return n;

        // Step 2
        for (int i = 0; i <= n; d[i, 0] = i++){}
        for (int j = 0; j <= m; d[0, j] = j++){}

        // Step 3
        for (int i = 1; i <= n; i++)
        {
            //Step 4
            for (int j = 1; j <= m; j++)
            {
                // Step 5
                int cost = (t[j-1] == s[i-1]) ? 0 : 1;

                // Step 6
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);

                // Step 7
                if (i > 1 && j > 1 && t[j - 1] == s[i - 2] && s[i - 1] == t[j - 2])
                {
                    d[i, j] = Math.Min(d[i, j], d[i - 2, j - 2] + cost);
                }
            }
        }

        // Step 8
        return d[n, m];
    }

    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    /// see also: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance
    /// default is case sensitive
    /// Levenshtein distance is a measure of the difference between two sequences.
    /// The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.
    /// Triangle inequality does not hold.
    /// One dimensional Array is used to store the distance matrix.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int OptimalDamerauLevenshteinDistance2(this string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[] d = new int[(n+1) * (m + 1)];

        // Step 1
        if (n == 0) return m;
        if (m == 0) return n;

        // Step 2
        for (int i = 0; i <= n; d[i*m+1] = i++) { }
        for (int j = 0; j <= m; d[j] = j++) { }

        // Step 3
        for (int i = 1; i <= n; i++)
        {
            //Step 4
            for (int j = 1; j <= m; j++)
            {
                // Step 5
                int cost = (t[j - 1] == s[i - 1]) ? 0 : 1;

                // Step 6
                d[i*(m+1)+j] = Math.Min(
                    Math.Min(d[(i - 1)*(m+1)+j] + 1, d[i*(m+1)+j - 1] + 1),
                    d[(i - 1) * (m + 1) + j - 1] + cost);

                // Step 7
                if (i > 1 && j > 1 && t[j - 1] == s[i - 2] && s[i - 1] == t[j - 2])
                {
                    d[i * (m + 1) + j] = Math.Min(d[i * (m + 1) + j], d[(i - 2) * (m + 1) + j - 2] + cost);
                }
            }
        }

        // Step 8
        return d[n * (m + 1) + m];
    }

    /// <summary>
    /// Knuth–Morris–Pratt algorithm for string matching.
    /// https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
    /// string matching algorithm that finds the first occurrence of a pattern in a text.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static (bool found, int? indice) KMPIndexOf(this string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[] pi = new int[m];
        int i = 0;
        int j = 0;
        while (i < n)
        {
            if (t[j] == s[i])
            {
                j++;
                i++;
            }
            if (j == m)
            {
                return (true, i - j);
            }
            else if (i < n && t[j] != s[i])
            {
                if (j != 0)
                {
                    j = pi[j - 1];
                }
                else
                {
                    i++;
                }
            }
        }
        return (false, null);
    }

    /// <summary>
    /// Knuth–Morris–Pratt algorithm for string matching.
    /// https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
    /// string matching algorithm that finds the first occurrence of a pattern in a text.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int KMPIndexOf2(this string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[] pi = new int[m];
        int i = 0;
        int j = 0;
        while (i < n)
        {
            if (t[j] == s[i])
            {
                j++;
                i++;
            }
            if (j == m)
            {
                return i - j;
            }
            else if (i < n && t[j] != s[i])
            {
                if (j != 0)
                {
                    j = pi[j - 1];
                }
                else
                {
                    i++;
                }
            }
        }
        return -1;
    }

    /// <summary>
    /// Knuth–Morris–Pratt algorithm for string matching.
    /// https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm
    /// string matching algorithm that finds the first occurrence of a pattern in a text.
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static bool TryGetKMP(this string s, string t, out int? indice)
    {
        int n = s.Length;
        int m = t.Length;
        int[] pi = new int[m];
        int i = 0;
        int j = 0;
        while (i < n)
        {
            if (t[j] == s[i])
            {
                j++;
                i++;
            }
            if (j == m)
            {
                indice = i - j;
                return true;
            }
            else if (i < n && t[j] != s[i])
            {
                if (j != 0)
                {
                    j = pi[j - 1];
                }
                else
                {
                    i++;
                }
            }
        }
        indice = null;
        return false;
    }
}


public static class StringExtensions
{
    public static LineSplitEnumerator SplitLines(this string str, char c1, char c2)
    {
        // LineSplitEnumerator is a struct so there is no allocation here
        return new LineSplitEnumerator(str.AsSpan(),c1,c2);
    }

    // Must be a ref struct as it contains a ReadOnlySpan<char>
    public ref struct LineSplitEnumerator
    {
        private char CARRIAGE_RETURN = '\r';
        private char NEXT_LINE = '\n';
        private ReadOnlySpan<char> _str;

        public LineSplitEnumerator(ReadOnlySpan<char> str, char c1, char c2)
        {
            CARRIAGE_RETURN = c1;
            NEXT_LINE = c2;
            _str = str;
            Current = default;
        }

        // Needed to be compatible with the foreach operator
        public LineSplitEnumerator GetEnumerator() => this;

        public bool MoveNext()
        {
            var span = _str;
            if (span.Length == 0) // Reach the end of the string
                return false;

            var index = span.IndexOfAny(CARRIAGE_RETURN, NEXT_LINE);
            if (index == -1) // The string is composed of only one line
            {
                _str = ReadOnlySpan<char>.Empty; // The remaining string is an empty string
                Current = new LineSplitEntry(span, ReadOnlySpan<char>.Empty);
                return true;
            }

            if (index < span.Length - 1 && span[index] == CARRIAGE_RETURN)
            {
                // Try to consume the '\n' associated to the '\r'
                var next = span[index + 1];
                if (next == NEXT_LINE)
                {
                    Current = new LineSplitEntry(span[..index], span.Slice(index, 2));
                    _str = span[(index + 2)..];
                    return true;
                }
            }

            Current = new LineSplitEntry(span[..index], span.Slice(index, 1));
            _str = span[(index + 1)..];
            return true;
        }

        public LineSplitEntry Current { get; private set; }
    }

    public readonly ref struct LineSplitEntry
    {
        public LineSplitEntry(ReadOnlySpan<char> line, ReadOnlySpan<char> separator)
        {
            Line = line;
            Separator = separator;
        }

        public ReadOnlySpan<char> Line { get; }
        public ReadOnlySpan<char> Separator { get; }

        // This method allow to deconstruct the type, so you can write any of the following code
        // foreach (var entry in str.SplitLines()) { _ = entry.Line; }
        // foreach (var (line, endOfLine) in str.SplitLines()) { _ = line; }
        // https://docs.microsoft.com/en-us/dotnet/csharp/deconstruct?WT.mc_id=DT-MVP-5003978#deconstructing-user-defined-types
        public void Deconstruct(out ReadOnlySpan<char> line, out ReadOnlySpan<char> separator)
        {
            line = Line;
            separator = Separator;
        }

        // This method allow to implicitly cast the type into a ReadOnlySpan<char>, so you can write the following code
        // foreach (ReadOnlySpan<char> entry in str.SplitLines())
        public static implicit operator ReadOnlySpan<char>(LineSplitEntry entry) => entry.Line;
    }
}