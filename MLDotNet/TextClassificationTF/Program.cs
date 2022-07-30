using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

string modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");
MLContext mlContext = new MLContext();
IDataView lookupMap = mlContext.Data.LoadFromTextFile(Path.Combine(modelPath, "imdb_word_index.csv"),
    columns: new[]
        {
            new TextLoader.Column("Words", DataKind.String, 0),
            new TextLoader.Column("Ids", DataKind.Int32, 1),
        },
    separatorChar: ','
);

Action<VariableLength, FixedLength> ResizeFeaturesAction = (s, f) =>
{
    int[] features = s.VariableLengthFeatures;
    Array.Resize(ref features, Config.FeatureLength);
    f.Features = features;
};
TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(modelPath);
DataViewSchema schema = tensorFlowModel.GetModelSchema();

Console.WriteLine(" ======================================================= ");
Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
VectorDataViewType featuresType = (VectorDataViewType)schema["Features"].Type;
Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
VectorDataViewType predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;
Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");
Console.WriteLine(" ======================================================= ");
IEstimator<ITransformer> pipeline =
    // Split the text into individual words
    mlContext.Transforms.Text.TokenizeIntoWords("TokenizedWords", "ReviewText")
        .Append(mlContext.Transforms.Conversion.MapValue(
            outputColumnName: "VariableLengthFeatures",
            lookupMap: lookupMap,
            keyColumn: lookupMap.Schema["Words"],
            valueColumn: lookupMap.Schema[name: "Ids"],
            inputColumnName: "TokenizedWords"))
        .Append(mlContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
        .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
        .Append(mlContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<MovieReview>());
ITransformer model = pipeline.Fit(dataView);
PredictSentiment(mlContext, model);

static void PredictSentiment(MLContext mlContext, ITransformer model)
{
    using PredictionEngine<MovieReview, MovieReviewSentimentPrediction> engine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);
    MovieReview review = new MovieReview()
    {
        ReviewText = "this film is really CRAP! I hated it! I'll never watch it again! The acting was so bad, the direction was terrible, and the story was too long. I'll never watch a film by that director again!",
    };
    MovieReviewSentimentPrediction sentimentPrediction = engine.Predict(review);
    Console.WriteLine("Number of classes: {0}", sentimentPrediction.Prediction.Length);
    Console.WriteLine("Is sentiment/review positive? {0}", sentimentPrediction.Prediction[1] > 0.5 ? "Yes." : "No.");
}

public class MovieReview
{
    public string ReviewText { get; set; }
}

/// <summary>
/// Class to hold the variable length feature vector. Used to define the
/// column names used as input to the custom mapping action.
/// </summary>
public class VariableLength
{
    /// <summary>
    /// This is a variable length vector designated by VectorType attribute.
    /// Variable length vectors are produced by applying operations such as 'TokenizeWords' on strings
    /// resulting in vectors of tokens of variable lengths.
    /// </summary>
    [VectorType]
    public int[] VariableLengthFeatures { get; set; }
}

/// <summary>
/// Class to hold the fixed length feature vector. Used to define the
/// column names used as output from the custom mapping action,
/// </summary>
public class FixedLength
{
    /// <summary>
    /// This is a fixed length vector designated by VectorType attribute.
    /// </summary>
    [VectorType(Config.FeatureLength)]
    public int[] Features { get; set; }
}

/// <summary>
/// Class to contain the output values from the transformation.
/// </summary>
public class MovieReviewSentimentPrediction
{
    [VectorType(2)]
    public float[] Prediction { get; set; }
}

static class Config
{
    public const int FeatureLength = 600;
}