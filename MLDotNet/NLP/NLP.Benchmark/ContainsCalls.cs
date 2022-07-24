// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using NLP.Extensions;

[MemoryDiagnoser]
[SimpleJob(BenchmarkDotNet.Jobs.RuntimeMoniker.Net60)]
[RPlotExporter]
public class ContainsCalls
{
    [Benchmark]
    public void Test()
    {
        TEST.Contains("123");
    }

    public const string TEST =
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTes123tTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest" +
        "TestTestTestTestTestTestTestTestTestTestTestTestTest";
}