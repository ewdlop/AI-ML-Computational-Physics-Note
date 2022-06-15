namespace NLP.Extensions;

public static partial class EnumerableExtension
{
    public static IEnumerable<(T Item, float probability)> Softmax<T>(this IEnumerable<T> collection, Func<T, float> scoreSelector)
    {
        double maxScore = collection.Max(scoreSelector);
        double sum = collection.Sum(r => Math.Exp(scoreSelector(r)) - maxScore);
        return collection.Select(r => (r, (float)(Math.Exp(scoreSelector(r) - maxScore) / sum)));
    }
}