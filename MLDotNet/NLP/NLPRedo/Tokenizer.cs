using Microsoft.ML.Data;

namespace NLPRedo;

public class Tokenizer
{
    private const string COMMON_DELIMETERS = ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'";
    private const string PREFIX_LABEL = "##";
    private const string CARRIAGE_RETURN_NEXT_LINE = "\r\n";
    private const string WHITE_SPACE = " ";
    private const string DOUBLE_WHITE_SPACE = "   ";
    private static readonly string PREFIX_LABEL_REPLACE = string.Empty;
    private readonly List<string> _vocabulary;
    private static readonly string[] Separator = new string[] { WHITE_SPACE, DOUBLE_WHITE_SPACE, CARRIAGE_RETURN_NEXT_LINE
    public Tokenizer(List<string> vocabulary)
    {
        _vocabulary = vocabulary;
    }

    public List<(string Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
    {
        IEnumerable<string> tokens = new string[] { Tokens.Classification };

        for (int i = 0; i < texts.Length; i++)
        {
            string text = texts[i];
            tokens = tokens.Concat(TokenizeSentence(text));
            tokens = tokens.Concat(new string[] { Tokens.Separation });
        }

        List<(string Token, int VocabularyIndex)> tokenAndIndex = tokens
            .SelectMany(TokenizeSubwords)
            .ToList();

        IEnumerable<long> segmentIndexes = SegmentIndex(tokenAndIndex);

        return tokenAndIndex.Zip(segmentIndexes,
            (tokenindex, segmentindex)
                => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex))
            .ToList();
    }

    public static List<string> Untokenize(List<string> tokens)
    {
        string currentToken = string.Empty;
        List<string> untokens = new();
        tokens.Reverse();

        tokens.ForEach(token =>
        {
            if (token.StartsWith(PREFIX_LABEL))
            {
                currentToken = $"{token.Replace(PREFIX_LABEL, PREFIX_LABEL_REPLACE)}{currentToken}";
            }
            else
            {
                currentToken = $"{token}{currentToken}";
                untokens.Add(currentToken);
                currentToken = string.Empty;
            }
        });

        untokens.Reverse();

        return untokens;
    }

    public static IEnumerable<long> SegmentIndex(List<(string token, int index)> tokens)
    {
        int segmentIndex = 0;
        List<long> segmentIndexes = new List<long>();

        for (int i = 0; i < tokens.Count; i++)
        {
            segmentIndexes.Add(segmentIndex);

            if (tokens[i].token == Tokens.Separation)
            {
                segmentIndex++;
            }
        }

        return segmentIndexes;
    }

    private IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
    {
        if (_vocabulary.Contains(word))
        {
            return new (string, int)[] { (word, _vocabulary.IndexOf(word)) };
        }

        List<(string, int)> tokens = new();
        string remaining = word;

        while (!string.IsNullOrEmpty(remaining) && remaining.Length > PREFIX_LABEL.Length)
        {
            string? prefix = _vocabulary.Where(remaining.StartsWith)
                .OrderByDescending(o => o.Length)
                .FirstOrDefault();

            if (prefix is null)
            {
                tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));

                return tokens;
            }

            remaining = remaining.Replace(prefix, PREFIX_LABEL);

            tokens.Add((prefix, _vocabulary.IndexOf(prefix)));
        }

        if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
        {
            tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
        }
        return tokens;
    }

    private static IEnumerable<string> TokenizeSentence(string text)
    {
        // remove spaces and split the , . : ; etc..
        return text.Split(Separator, StringSplitOptions.None)
            .SelectMany(o => o.SplitAndKeep(COMMON_DELIMETERS.ToArray()))
            .Select(o => o.ToLower());
    }
}