using System.Collections.Generic;

namespace InformationRetrieval;

public static class DamerauLevenshteinDistance
{
    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// <para><see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance"></see></para>
    /// <para>See also: <see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance"></see></para>
    /// <para>Default is case sensitive</para>
    /// <para>Levenshtein distance is a measure of the difference between two sequences.</para>
    /// <para>The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.</para>
    /// <para>Triangle inequality does not hold.</para>
    /// <para>One dimensional Array is used to store the distance matrix.</para>
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int OptimalDamerauLevenshteinDistance(this string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        int[] d = new int[(n + 1) * (m + 1)];

        // Step 1
        if (n == 0) return m;
        if (m == 0) return n;

        // Step 2
        for (int i = 0; i <= n; d[i * m + 1] = i++) { }
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
                d[i * (m + 1) + j] = Math.Min(
                    Math.Min(d[(i - 1) * (m + 1) + j] + 1, // deletion
                             d[i * (m + 1) + j - 1] + 1), // insertion
                             d[(i - 1) * (m + 1) + j - 1] + cost);  // substitution

                // Step 7
                if (i > 1 && j > 1 && t[j - 1] == s[i - 2] && s[i - 1] == t[j - 2])
                {
                    d[i * (m + 1) + j] = Math.Min(d[i * (m + 1) + j],
                                                  d[(i - 2) * (m + 1) + j - 2] + cost); // transposition
                }
            }
        }

        // Step 8
        return d[n * (m + 1) + m];
    }
    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// <para><see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance"></see></para>
    /// <para>See also: <see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance"></see></para>
    /// <para>Default is case sensitive</para>
    /// <para>Levenshtein distance is a measure of the difference between two sequences.</para>
    /// <para>The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.</para>
    /// <para>Triangle inequality does not hold.</para>
    /// <para>One dimensional Array is used to store the distance matrix.</para>
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int OptimalDamerauLevenshteinDistance(this ReadOnlySpan<char> s, ReadOnlySpan<char> t)
    {
        int n = s.Length;
        int m = t.Length;
        int[] d = new int[(n + 1) * (m + 1)];

        // Step 1
        if (n == 0) return m;
        if (m == 0) return n;

        // Step 2
        for (int i = 0; i <= n; d[i * m + 1] = i++) { }
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
                d[i * (m + 1) + j] = Math.Min(
                    Math.Min(d[(i - 1) * (m + 1) + j] + 1, // deletion
                             d[i * (m + 1) + j - 1] + 1), // insertion
                             d[(i - 1) * (m + 1) + j - 1] + cost);  // substitution

                // Step 7
                if (i > 1 && j > 1 && t[j - 1] == s[i - 2] && s[i - 1] == t[j - 2])
                {
                    d[i * (m + 1) + j] = Math.Min(d[i * (m + 1) + j],
                                                  d[(i - 2) * (m + 1) + j - 2] + cost); // transposition
                }
            }
        }

        // Step 8
        return d[n * (m + 1) + m];
    }
    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// <para><see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance"></see></para>
    /// <para>See also: <see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance"></see></para>
    /// <para>Default is case sensitive</para>
    /// <para>Levenshtein distance is a measure of the difference between two sequences.</para>
    /// <para>The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.</para>
    /// <para>Triangle inequality does not hold.</para>
    /// <para>One dimensional Array is used to store the distance matrix.</para>
    /// <para>Use stack allocation</para>
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int OptimalDamerauLevenshteinDistanceStackAlloc(this string s, string t)
    {
        int n = s.Length;
        int m = t.Length;
        Span<int> d = stackalloc int[(n + 1) * (m + 1)];

        // Step 1
        if (n == 0) return m;
        if (m == 0) return n;

        // Step 2
        for (int i = 0; i <= n; d[i * m + 1] = i++) { }
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
                d[i * (m + 1) + j] = Math.Min(
                    Math.Min(d[(i - 1) * (m + 1) + j] + 1, // deletion
                             d[i * (m + 1) + j - 1] + 1), // insertion
                             d[(i - 1) * (m + 1) + j - 1] + cost);  // substitution

                // Step 7
                if (i > 1 && j > 1 && t[j - 1] == s[i - 2] && s[i - 1] == t[j - 2])
                {
                    d[i * (m + 1) + j] = Math.Min(d[i * (m + 1) + j],
                                                  d[(i - 2) * (m + 1) + j - 2] + cost); // transposition
                }
            }
        }

        // Step 8
        return d[n * (m + 1) + m];
    }
    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// <para><see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance"></see></para>
    /// <para>See also: <see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance"></see></para>
    /// <para>Default is case sensitive</para>
    /// <para>Levenshtein distance is a measure of the difference between two sequences.</para>
    /// <para>The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.</para>
    /// <para>Triangle inequality does not hold.</para>
    /// <para>One dimensional Array is used to store the distance matrix.</para>
    /// <para>Use stack allocation</para>
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    /// <summary>
    /// Optimal Damerau–Levenshtein distance.
    /// <para><see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance"></see></para>
    /// <para>See also: <see href="https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance"></see></para>
    /// <para>Default is case sensitive</para>
    /// <para>Levenshtein distance is a measure of the difference between two sequences.</para>
    /// <para>The number of edit operations needed to make the strings equal under the condition that no substring is edited more than once.</para>
    /// <para>Triangle inequality does not hold.</para>
    /// <para>One dimensional Array is used to store the distance matrix.</para>
    /// <para>Use stack allocation</para>
    /// </summary>
    /// <param name="s"></param>
    /// <param name="t"></param>
    /// <returns></returns>
    public static int OptimalDamerauLevenshteinDistanceStackAlloc(this ReadOnlySpan<char> s, ReadOnlySpan<char> t)
    {
        int n = s.Length;
        int m = t.Length;
        Span<int> d = stackalloc int[(n + 1) * (m + 1)];

        // Step 1
        if (n == 0) return m;
        if (m == 0) return n;

        // Step 2
        for (int i = 0; i <= n; d[i * m + 1] = i++) { }
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
                d[i * (m + 1) + j] = Math.Min(
                    Math.Min(d[(i - 1) * (m + 1) + j] + 1, // deletion
                             d[i * (m + 1) + j - 1] + 1), // insertion
                             d[(i - 1) * (m + 1) + j - 1] + cost);  // substitution

                // Step 7
                if (i > 1 && j > 1 && t[j - 1] == s[i - 2] && s[i - 1] == t[j - 2])
                {
                    d[i * (m + 1) + j] = Math.Min(d[i * (m + 1) + j],
                                                  d[(i - 2) * (m + 1) + j - 2] + cost); // transposition
                }
            }
        }

        // Step 8
        return d[n * (m + 1) + m];
    }
    public static bool OptimalDamerauLevenshteinDistanceThreshold(this string s, string t, int threshold)
    {
        return OptimalDamerauLevenshteinDistance(s, t) <= threshold;
    }
}
