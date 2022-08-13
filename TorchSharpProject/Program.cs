using TorchSharp;
using TorchSharp.Modules;
using TorchSharpProject.Modules;

Test2();

static void Random()
{
    Bernoulli bernolli = new(torch.tensor(0.5f));
    Console.WriteLine(bernolli.sample().item<float>());
}

static void Test2()
{
    float learngRate = 0.01f;
    using Trivial model = new Trivial();
    Loss loss = torch.nn.functional.mse_loss();
    List<torch.Tensor> data = Enumerable.Range(0, 16).Select(_ => torch.rand(32, 1000)).ToList();
    List<torch.Tensor> results = Enumerable.Range(0, 16).Select(_ => torch.rand(32, 10)).ToList();
    using SGD optimizer = torch.optim.SGD(model.parameters(), learngRate);
    torch.optim.lr_scheduler.LRScheduler scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.95);

    for (int i = 0; i < 300; i++)
    {
        for (int idx = 0; i < data.Count; i++)
        {
            // Compute the loss
            using torch.Tensor output = loss(model.forward(data[idx]), results[idx]);

            // Clear the gradients before doing the back-propagation
            model.zero_grad();

            // Do back-progatation, which computes all the gradients.
            output.backward();

            optimizer.step();
        }
        scheduler.step();
    }

    Console.WriteLine(loss(model.forward(data[0]), results[0]).item<float>());
}


static void Test()
{
    using Linear linear = torch.nn.Linear(1000, 100);
    using Linear linear2 = torch.nn.Linear(100, 10);
    using Sequential seq = torch.nn.Sequential(
        ("lin1", linear),
        ("relu1", torch.nn.ReLU()),
        ("drop1", torch.nn.Dropout(0.1)),
        ("lin2", linear2));
    using (DisposeScope d = torch.NewDisposeScope())
    {
        torch.Tensor x = torch.rand(64, 1000);
        torch.Tensor y = torch.rand(64, 10);
        using Adam optimizer = torch.optim.Adam(seq.parameters());
        Loss loss = torch.nn.functional.mse_loss(torch.nn.Reduction.Sum);

        for (int i = 0; i < 10; i++)
        {
            torch.Tensor eval = seq.forward(x);
            torch.Tensor output = loss(eval, y);

            optimizer.zero_grad();

            output.backward();

            optimizer.step();
        }
    }
}