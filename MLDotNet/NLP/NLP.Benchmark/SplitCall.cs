// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using NLP.Extensions;

[MemoryDiagnoser]
[SimpleJob(BenchmarkDotNet.Jobs.RuntimeMoniker.Net60)]
public class SplitCall
{
    private const string Data = "Nickname: meziantou\r\nFirstName: Gérald\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré\nLastName: Barré";

    [Benchmark]
    public void Test()
    {
        foreach (var line in Data.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
        {
        }
    }
    
    [Benchmark]
    public void Test2()
    {

        foreach (ReadOnlySpan<char> item in Data.SplitLines('\r','\n'))
        {
        }
    }
}