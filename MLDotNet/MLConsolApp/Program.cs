using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

string test = """
Sentiment   SentimentText
1 Stop trolling, zapatancas, calling me a liar merely demonstartes that you arer Zapatancas. You may choose to chase every legitimate editor from this site and ignore me but I am an editor with a record that isnt 99% trolling and therefore my wishes are not to be completely ignored by a sockpuppet like yourself. The consensus is overwhelmingly against you and your trollin g lover Zapatancas,  
1 ::::: Why are you threatening me? I'm not being disruptive, its you who is being disruptive.   
0 " *::Your POV and propaganda pushing is dully noted. However listing interesting facts in a netral and unacusitory tone is not POV. You seem to be confusing Censorship with POV monitoring. I see nothing POV expressed in the listing of intersting facts. If you want to contribute more facts or edit wording of the cited fact to make them sound more netral then go ahead. No need to CENSOR interesting factual information. "
0 ::::::::This is a gross exaggeration. Nobody is setting a kangaroo court. There was a simple addition concerning the airline. It is the only one disputed here.   
""";
MLContext mlContext = new();
TextLoader loader = mlContext.Data.CreateTextLoader(new[] {
    new TextLoader.Column("IsToxic", DataKind.Boolean, 0),
    new TextLoader.Column("Message", DataKind.String, 1)
}, hasHeader: true);

IDataView data = loader.Load(test);

string[] messageTexts = data.GetColumn<string>(data.Schema["Message"]).Take(20).ToArray();

var pipeline =
    mlContext.Transforms.Text.FeaturizeText("TextFeatures", "Message")
     .Append(mlContext.Transforms.Text.NormalizeText("NormalizedMessage", "Message"))
     .Append(mlContext.Transforms.Text.ProduceWordBags("BagOfWords", "NormalizedMessage"))
     .Append(mlContext.Transforms.Text.ProduceHashedWordBags("BagOfBigrams", "NormalizedMessage",
                ngramLength: 2, useAllLengths: false))
    .Append(mlContext.Transforms.Text.TokenizeIntoCharactersAsKeys("MessageChars", "Message"))
    .Append(mlContext.Transforms.Text.ProduceNgrams("BagOfTrichar", "MessageChars",
                ngramLength: 3, weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf)
    .Append(mlContext.Transforms.Text.TokenizeIntoWords("TokenizedMessage", "NormalizedMessage"))
    .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Embeddings", "TokenizedMessage",
                WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding)));




IDataView transformedData = pipeline.Fit(data).Transform(data);

var embeddings = transformedData.GetColumn<float[]>("Embeddings").Take(10).ToArray();
var unigrams = transformedData.GetColumn<float[]>("BagOfWords").Take(10).ToArray();

