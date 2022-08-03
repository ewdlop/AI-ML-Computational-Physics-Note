using NLP.Extensions;

namespace NLPTest;

public class UnitTest1
{
    [Fact]
    public void Test1()
    {
        string text = "This is a test.This is a test.";
        IEnumerable<ReadOnlyMemory<char>> text2 = text.AsSplitMemoryAndNotKeepingDelimters('.');
        Assert.Equal(3, text2.Count());
        Assert.Equal("This is a test", text2.First().ToString());
        Assert.Equal("This is a test", text2.Skip(1).First().ToString());
        Assert.Equal("", text2.Last().ToString());

        IEnumerable<ReadOnlyMemory<char>> text3 = text.AsSplitMemoryKeepingDelimter('.');
        Assert.Equal(4, text3.Count());
        Assert.Equal("This is a test", text3.First().ToString());
        Assert.Equal(".", text3.Skip(1).First().ToString());
        Assert.Equal("This is a test", text3.Skip(2).First().ToString());
        Assert.Equal(".", text3.Last().ToString());

        string text4 = "This   is a test.  This is a   test.";
        Assert.DoesNotContain(" ", text4.AsTokenizeSentence().Select(s => s.ToString()));

        ReadOnlyMemory<char> read = text4.AsMemory()[text4.Length..];
        Assert.True(read.IsEmpty);

    }

    [Fact]
    public void Test2()
    {
        Assert.Equal(3, "CA".OptimalDamerauLevenshteinDistance("ABC"));
        Assert.Equal(3, "CA".OptimalDamerauLevenshteinDistance2("ABC"));
    }
}