using NLP.Extensions;
using System.Text;

namespace NLP.BERT.Tokenizers;

public class Tokenizer
{
    public const string PREFIX_MARK = "##";
    private readonly IList<ReadOnlyMemory<char>> _vocabulary;
    public Tokenizer(IList<ReadOnlyMemory<char>> vocabulary)
    {
        _vocabulary = vocabulary;
    }

    public List<(ReadOnlyMemory<char> Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
    {
        List<ReadOnlyMemory<char>> tokens = new List<ReadOnlyMemory<char>>(texts.Length);

        for (int i = 0; i < texts.Length; i++)
        {
            string text = texts[i];
            tokens.AddRange(text.AsTokenizeSentence());
            tokens.Add(Tokens.Separation);
        }

        IEnumerable<(ReadOnlyMemory<char> Token, int VocabularyIndex)> tokenAndIndex = tokens
            .SelectMany(TokenizeSubwords2);

        IEnumerable<long> segmentIndexes = tokenAndIndex.ToSegmentIndices();

        return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex)
                            => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToList();
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

    private List<(ReadOnlyMemory<char> Token, int VocabularyIndex)> TokenizeSubwords2(ReadOnlyMemory<char> word)
    {
        if (_vocabulary.Contains(word))
        {
            return new List<(ReadOnlyMemory<char>, int)> { (word, _vocabulary.IndexOf(word)) };
        }
        List<(ReadOnlyMemory<char> Token, int VocabularyIndex)> tokens = new();
        ReadOnlyMemory<char> remaining = word;
        while (!word.IsEmpty && remaining.Length > PREFIX_MARK.Length)
        {
            (ReadOnlyMemory<char> prefix, int index) = _vocabulary
                    .Where(x=>remaining.Span.StartsWith(x.Span))
                    .Select((p, i) => (p, i))
                    .OrderByDescending(o => o.p.Length)
                    .FirstOrDefault();
            if (prefix.Length == 0)
            {
                tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
                return tokens;
            }

            remaining = new StringBuilder(remaining.ToString())
                .Replace(prefix.ToString(), PREFIX_MARK).ToString().AsMemory();
            tokens.Add((prefix, index));
        }

        if (word.IsEmpty && !tokens.Any())
        {
            tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
        }
        return tokens;
    }

    private List<(ReadOnlyMemory<char> Token, int VocabularyIndex)> TokenizeSubwordsWithMarshalling(ReadOnlyMemory<char> word)
    {
        if (_vocabulary.Contains(word))
        {
            return new List<(ReadOnlyMemory<char>, int)> { (word, _vocabulary.IndexOf(word)) };
        }
        List<(ReadOnlyMemory<char> Token, int VocabularyIndex)> tokens = new();
        ReadOnlyMemory<char> remaining = word;
        while (!word.IsEmpty && remaining.Length > PREFIX_MARK.Length)
        {
            (ReadOnlyMemory<char> prefix, int index) = _vocabulary.Where(v => v.Span.StartsWith(word.Span))
                    .Select((p, i) => (p, i))
                    .OrderByDescending(o => o.p.Length)
                    .FirstOrDefault();
            if (prefix.Length == 0)
            {
                tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
                return tokens;
            }

            if (System.Runtime.InteropServices.MemoryMarshal.TryGetString(remaining[prefix.Length..], out string? remainingString, out int start, out int length))
            {
                remaining = remainingString.Replace(prefix.ToString(), "##").ToString().AsMemory();
            }

            tokens.Add((prefix, index));
        }

        if (word.IsEmpty && !tokens.Any())
        {
            tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
        }
        return tokens;
    }
}
