using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

//https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/diffusion.py
public class TimeEmbedding(string Name, int NEmbedding) : Module(Name)
{
    public Linear Linear1 => Linear(NEmbedding, 4 * NEmbedding);
    public Linear Linear2 => Linear(4 * NEmbedding, 4 * NEmbedding);

    public Tensor Forward(Tensor x)
    {
        // x: (1, 320)
        
        // (1, 320) -> (1, 1280)
        
        var result = Linear1.forward(x);
        
        // (1, 1280) -> (1, 1280)
        
        result = functional.silu(result); ;

        // (1, 1280) -> (1, 1280)
        result = Linear2.forward(x);

        return result;
    }
}