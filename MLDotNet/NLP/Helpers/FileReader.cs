namespace NLP.Helpers;

public static class FileReaderExtension
{
    public static async IAsyncEnumerable<ReadOnlyMemory<char>> ReadLinesAsMemoryAsync(this string filename)
    {
        using StreamReader reader = new(filename);
        string? line;
        while ((line = await reader.ReadLineAsync()) is not null)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                yield return line.AsMemory();
            }
        }
    }

    public static async IAsyncEnumerable<string> ReadLinesAsync(this string filename)
    {
        using StreamReader reader = new(filename);
        string? line;

        while ((line = await reader.ReadLineAsync()) is not null)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                yield return line;
            }
        }
    }

    public static IEnumerable<string> ReadLines(this string filename)
    {
        using StreamReader reader = new(filename);
        string? line;

        while ((line = reader.ReadLine()) is not null)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                yield return line;
            }
        }
    }
}