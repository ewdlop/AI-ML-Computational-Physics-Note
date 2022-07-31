using Microsoft.ML;
using Microsoft.ML.Data;


string assetsPath = Path.Combine(Environment.CurrentDirectory, "Assets");
string imagesFolder = Path.Combine(assetsPath, "Images");
string trainTagsTsv = Path.Combine(imagesFolder, "tags.tsv");
string testTagsTsv = Path.Combine(imagesFolder, "test-tags.tsv");
string predictSingleImage = Path.Combine(imagesFolder, "toaster3.jpg");
string inceptionTensorFlowModel = Path.Combine(assetsPath, "Inception", "tensorflow_inception_graph.pb");

MLContext mlContext = new MLContext();
IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: testTagsTsv, hasHeader: false);
ITransformer model = GenerateModel(mlContext);
ClassifySingleImage(mlContext, model);

ITransformer GenerateModel(MLContext mlContext)
{
    // L-BFGS
    //https://en.wikipedia.org/wiki/Limited-memory_BFGS
    //Broyden–Fletcher–Goldfarb–Shanno algorithm
    //https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "input",
                    imageWidth: InceptionSettings.ImageWidth,
                    imageHeight: InceptionSettings.ImageHeight,
                    inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input",
                    interleavePixelColors: InceptionSettings.ChannelsLast,
                    offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(inceptionTensorFlowModel).
                    ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2_pre_activation" },
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelKey", 
                    inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                    labelColumnName: "LabelKey",
                    featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabelValue", 
                    inputColumnName: "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

    IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: trainTagsTsv, hasHeader: false);
    //The Fit() method trains your model by applying the training dataset to the pipeline
    ITransformer model = pipeline.Fit(trainingData);
    IDataView predictions = model.Transform(testData);
    // Create an IEnumerable for the predictions for displaying results
    IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    DisplayResults(imagePredictionData);
    MulticlassClassificationMetrics metrics =
        mlContext.MulticlassClassification.Evaluate(predictions,
            labelColumnName: "LabelKey",
            predictedLabelColumnName: "PredictedLabel");
    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {string.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
    return model;
}
void ClassifySingleImage(MLContext mlContext, ITransformer model)
{
    ImageData imageData = new()
    {
        ImagePath = predictSingleImage
    };

    // Make prediction function (input = ImageData, output = ImagePrediction)
    using PredictionEngine<ImageData, ImagePrediction> predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    ImagePrediction prediction = predictor.Predict(imageData);
    Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
}
void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (ImagePrediction prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
    }
}
struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
}
public class ImagePrediction : ImageData
{
    public float[] Score;

    public string PredictedLabelValue;
}
public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath;

    [LoadColumn(1)]
    public string Label;
}