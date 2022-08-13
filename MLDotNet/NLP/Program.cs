//https://rubikscode.net/2021/04/19/machine-learning-with-ml-net-nlp-with-bert/

using NLP.BERT;
using System.Text.Json;

//IReadonlyMemory implmeent doesnt work 

string context = "Jim is walking throught the woods.";
string question = "What is his name?";
BidirectionalEncoderRepresentationsFromTransformers model = 
    new BidirectionalEncoderRepresentationsFromTransformers("bertsquad-10.onnx", 30522);

await model.ReadVocabularyFilePAsync("vocab.txt");

(List<string> tokens, float probability) = model.Predict(context, question);
Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));