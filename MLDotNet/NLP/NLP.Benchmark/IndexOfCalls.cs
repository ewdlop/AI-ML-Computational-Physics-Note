// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using NLP.Extensions;

[MemoryDiagnoser]
[SimpleJob(BenchmarkDotNet.Jobs.RuntimeMoniker.Net60)]
[RPlotExporter]
public class IndexOfCalls
{
    [Benchmark]
    public void Normal()
    {
        int index = TEST.IndexOf("123");
    }

    [Benchmark]
    public void IndexOf()
    {
        (bool found, int? indice) = TEST.KMPIndexOf("123");
    }

    [Benchmark]
    public void IndexOf2()
    {
         int indice = TEST.KMPIndexOf2("123");
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