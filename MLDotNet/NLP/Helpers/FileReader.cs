namespace NLP.Helpers;

public static class FileReader
{
    public static async IAsyncEnumerable<ReadOnlyMemory<char>> ReadFileAsync(string filename)
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
}