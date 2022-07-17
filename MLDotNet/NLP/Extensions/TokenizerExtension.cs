using NLP.BERT.Tokenizers;
using System.Text;

namespace NLP.Extensions;

public static class TokenizerExtension
{
    public static IEnumerable<long> ToSegmentIndices(this IEnumerable<(ReadOnlyMemory<char> token, int index)> tokens)
    {
        int segmentIndex = 0;
        List<long> segmentIndexes = new List<long>(tokens.Count());

        foreach ((ReadOnlyMemory<char> token, int index) in tokens)
        {
            segmentIndexes.Add(segmentIndex);

            if (token.Equals(Tokens.Separation))
            {
                segmentIndex++;
            }
        }

        return segmentIndexes;
    }
    
    public static List<string> ToUntokenizedString(this List<ReadOnlyMemory<char>> tokens)
    {
        StringBuilder currentTokenBuilder = new();
        List<string>? untokens = new List<string>();
        tokens.Reverse();

        foreach (ReadOnlyMemory<char> token in tokens)
        {
            if (token.Span.StartsWith("##"))
            {
                currentTokenBuilder.Insert(0, token.ToString().Replace("##", ""));
            }
            else
            {
                currentTokenBuilder.Insert(0, token.ToString().Replace("##", ""));
                untokens.Add(currentTokenBuilder.ToString());
                currentTokenBuilder.Clear();
            }
        }

        untokens.Reverse();

        return untokens;
    }

    public static List<string> ToUntokenizedStringWithMarShalling(this List<ReadOnlyMemory<char>> tokens)
    {
        StringBuilder currentTokenBuilder = new();
        List<string>? untokens = new List<string>();
        tokens.Reverse();

        foreach (ReadOnlyMemory<char> token in tokens)
        {
            if (token.Span.StartsWith("##"))
            {
                if(System.Runtime.InteropServices.MemoryMarshal.TryGetString(token, out string? tokenString, out int start, out int length))
                {
                    currentTokenBuilder.Insert(0, tokenString.Replace("##", ""));
                }
            }
            else
            {
                if (System.Runtime.InteropServices.MemoryMarshal.TryGetString(token, out string? tokenString, out int start, out int length))
                {
                    currentTokenBuilder.Insert(0, tokenString.Replace("##", ""));
                }
                untokens.Add(currentTokenBuilder.ToString());
                currentTokenBuilder.Clear();
            }
        }

        untokens.Reverse();

        return untokens;
    }
}