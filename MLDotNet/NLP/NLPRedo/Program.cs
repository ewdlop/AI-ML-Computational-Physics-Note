using NLPRedo;
using System.Text.Json;

Console.WriteLine("");

Bert model = new Bert("vocab.txt", "bertsquad-10.onnx");

var (tokens, probability) = model.Predict(
    "Jim is walking through the woods.",
    "What is his name?");

Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));
