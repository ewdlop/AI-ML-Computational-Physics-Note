namespace NLP.BERT.Tokenizers;

public class Tokens
{
    public static readonly ReadOnlyMemory<char> Padding = "".AsMemory();
    public static readonly ReadOnlyMemory<char> Unknown = "[UNK]".AsMemory();
    public static readonly ReadOnlyMemory<char> Classification = "[CLS]".AsMemory();
    public static readonly ReadOnlyMemory<char> Separation = "[SEP]".AsMemory();
    public static readonly ReadOnlyMemory<char> Mask = "[MASK]".AsMemory();
}
