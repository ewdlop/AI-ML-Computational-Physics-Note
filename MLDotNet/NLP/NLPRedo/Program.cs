using NLPRedo;
using System.Text.Json;

Console.WriteLine("");

BidirectionalEncoderRepresentationsFromTransformers model = new BidirectionalEncoderRepresentationsFromTransformers("vocab.txt", "bertsquad-10.onnx");

(List<string> tokens, float probability) = model.Predict(
    "Jack is walking through the woods. Jack has a gorgeous girlfriend named Mary,",
    "Who is his girlfriend?");

Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));
