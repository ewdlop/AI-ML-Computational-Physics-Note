using TorchSharp;

namespace TorchSharpProject.Modules;

public class ResNet : torch.nn.Module
{
    private TorchSharp.Modules.Sequential layers;

    public ResNet(string name,
                  Func<string, int, int, int, torch.nn.Module> block,
                  int expansion,
                  IList<int> num_blocks,
                  int numClasses,
                  torch.Device? device = null) 
        : base(nameof(Trivial))
    {
        //if (planes.Length != strides.Length) throw new ArgumentException("'planes' and 'strides' must have the same length.");

        List<(string, torch.nn.Module)> modules = new List<(string, torch.nn.Module)>();

        modules.Add(($"conv2d-first", torch.nn.Conv2d(3, 64, kernelSize: 3, stride: 1, padding: 1, bias: false)));
        modules.Add(($"bnrm2d-first", torch.nn.BatchNorm2d(64)));
        modules.Add(($"relu-first", torch.nn.ReLU(inPlace: true)));
        MakeLayer(modules, block, expansion, 64, num_blocks[0], 1);
        MakeLayer(modules, block, expansion, 128, num_blocks[1], 2);
        MakeLayer(modules, block, expansion, 256, num_blocks[2], 2);
        MakeLayer(modules, block, expansion, 512, num_blocks[3], 2);
        modules.Add(("avgpool", torch.nn.AvgPool2d(new long[] { 4, 4 })));
        modules.Add(("flatten", torch.nn.Flatten()));
        modules.Add(($"linear", torch.nn.Linear(512 * expansion, numClasses)));

        layers = torch.nn.Sequential(modules);
        RegisterComponents();

    }

    private void MakeLayer(List<(string, torch.nn.Module)> modules, Func<string, int, int, int, torch.nn.Module> block, int expansion, int v1, int v2, int v3)
    {
        throw new NotImplementedException();
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        return layers.forward(input);
    }
}