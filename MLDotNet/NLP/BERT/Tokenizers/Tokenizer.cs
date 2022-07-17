using NLP.Extensions;
using System.Text;

namespace NLP.BERT.Tokenizers;

public class Tokenizer
{
    private readonly IList<ReadOnlyMemory<char>> _vocabulary;
    public Tokenizer(IList<ReadOnlyMemory<char>> vocabulary)
    {
        _vocabulary = vocabulary;
    }

    public List<(ReadOnlyMemory<char> Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
    {
        List<ReadOnlyMemory<char>> tokens = new();

        foreach (string text in texts)
        {
            tokens.AddRange(text.AsTokenizeSentence());
            tokens.Add(Tokens.Separation);
        }

        IEnumerable<(ReadOnlyMemory<char> Token, int VocabularyIndex)> tokenAndIndex = tokens
            .SelectMany(TokenizeSubwords);

        IEnumerable<long> segmentIndexes = ToSegmentIndex(tokenAndIndex);

        return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex)
                            => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToList();
    }

    public static IEnumerable<long> ToSegmentIndex(IEnumerable<(ReadOnlyMemory<char> token, int index)> tokens)
    {
        int segmentIndex = 0;
        List<long> segmentIndexes = new List<long>();

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

    //not done, conflict with TokenizeSubwords
    public static List<string> Untokenize(List<ReadOnlyMemory<char>> tokens)
    {
        StringBuilder currentTokenBuilder = new();
        List<string>? untokens = new List<string>();
        tokens.Reverse();

        foreach (ReadOnlyMemory<char> token in tokens)
        {
            if (token.Span.StartsWith("##"))
            {
                currentTokenBuilder.Insert(0,token.ToString().Replace("##", ""));
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


    private List<(ReadOnlyMemory<char> Token, int VocabularyIndex)> TokenizeSubwords(ReadOnlyMemory<char> word)
    {
        if (_vocabulary.Contains(word))
        {
            return new List<(ReadOnlyMemory<char>, int)> { (word, _vocabulary.IndexOf(word)) };
        }
        List<(ReadOnlyMemory<char> Token, int VocabularyIndex)> tokens = new();
        ReadOnlyMemory<char> remaining = word;
        while (!word.IsEmpty && !remaining.IsEmpty)
        {
            (ReadOnlyMemory<char> prefix, int index) = _vocabulary.Where(v => v.Span.StartsWith(word.Span))
                    .Select((p,i)=>(p,i))
                    .OrderByDescending(o => o.p.Length)
                    .FirstOrDefault();
            if (prefix.Length == 0)
            {
                tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
                return tokens;
            }

            remaining = remaining[prefix.Length..];
            tokens.Add((prefix, index));
        }

        if (word.IsEmpty && !tokens.Any())
        {
            tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
        }
        return tokens;
    }
}