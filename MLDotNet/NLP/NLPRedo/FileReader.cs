namespace NLPRedo;

public static class FileReader
{
    public static List<string> ReadFile(string filename)
    {
        List<string> result = new();

        using (StreamReader reader = new(filename))
        {
            string? line;

            while ((line = reader.ReadLine()) is not null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    result.Add(line);
                }
            }
        }

        return result;
    }
}