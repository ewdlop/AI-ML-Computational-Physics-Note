using Microsoft.VisualBasic;
using System.Numerics;
using static TorchSharp.torch;

namespace TorchSharpProject.Modules;

public class Trivial : nn.Module
{
    public Trivial()
        : base(nameof(Trivial))
    {
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using Tensor x = lin1.forward(input);
        using Tensor y = nn.functional.relu(x);
        return lin2.forward(y);
    }

    private readonly nn.Module lin1 = nn.Linear(1000, 100);
    private readonly nn.Module lin2 = nn.Linear(100, 10);
}