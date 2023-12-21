using TorchSharp.Modules;

namespace TorchSharpSD;

//https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/diffusion.py
public class TimeEmbedding(string Name, int NEmbedding) : TorchSharp.torch.nn.Module(Name)
{
    public Linear Linear1 => TorchSharp.torch.nn.Linear(NEmbedding, 4 * NEmbedding);
    public Linear Linear2 => TorchSharp.torch.nn.Linear(4 * NEmbedding, 4 * NEmbedding);

    public TorchSharp.torch.Tensor Forward(TorchSharp.torch.Tensor x)
    {
        // x: (1, 320)
        
        // (1, 320) -> (1, 1280)
        
        var result = Linear1.forward(x);
        
        // (1, 1280) -> (1, 1280)
        
        result = TorchSharp.torch.nn.functional.silu(result); ;

        // (1, 1280) -> (1, 1280)
        result = Linear2.forward(result);

        return result;
    }
}


//pclass CLIPEmbedding(nn.Module) :
//    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
//        super().__init__()


//        self.token_embedding = nn.Embedding(n_vocab, n_embd)
//        # A learnable weight matrix encodes the position information for each token
//        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))


//    def forward(self, tokens):
//        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
//        x = self.token_embedding(tokens)
//        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
//        x += self.position_embedding

//public class CLIPEmbedding(string Name, int NVocabulary, int NEmbedding, int NToken) : TorchSharp.torch.nn.Module(Name)
//{
//    publi TokenEmbedding => em(NVocab, NEmb);
//    public PositionEmbedding => Parameter(torch.zeros((NToken, NEmb)));


//    public Tensor Forward(Tensor tokens)
//    {
//        // (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
//        var x = TokenEmbedding.forward(tokens);
//        // (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
//        x += PositionEmbedding;

//        return x;
//    }
//}