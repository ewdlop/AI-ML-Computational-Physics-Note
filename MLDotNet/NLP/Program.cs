//https://rubikscode.net/2021/04/19/machine-learning-with-ml-net-nlp-with-bert/

using NLP.BERT;
using System.Text.Json;

BidirectionalEncoderRepresentationsFromTransformers model = new BidirectionalEncoderRepresentationsFromTransformers(
                    "..\\BertMlNet\\Assets\\Model\\bertsquad-10.onnx");

await model.ReadVocabularyFilePAsync("..\\BertMlNet\\Assets\\Vocabulary\\vocab.txt");

(List<string> tokens, float probability) = model.Predict(args[0], args[1]);
Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));