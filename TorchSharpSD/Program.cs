using TorchSharp;
using static TorchSharp.torch.nn;

var lin1 = Linear(1000, 100);
var lin2 = Linear(100, 10);
var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

var x = torch.randn(64, 1000);
var y = torch.randn(64, 10);

var optimizer = torch.optim.Adam(seq.parameters());

for (int i = 0; i < 10; i++)
{
    var eval = seq.forward(x);
    var output = functional.mse_loss(eval, y, Reduction.Sum);

    optimizer.zero_grad();

    output.backward();

    optimizer.step();
}
