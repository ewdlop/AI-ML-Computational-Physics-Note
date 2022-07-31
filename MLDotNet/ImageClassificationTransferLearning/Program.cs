using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

string workspaceRelativePath = Path.Combine(Environment.CurrentDirectory, "workspace");
string assetsRelativePath = Path.Combine(Environment.CurrentDirectory, "Assets");
MLContext mlContext = new MLContext();
IEnumerable<ImageData> images = LoadImagesFromDirectory(
    folder: assetsRelativePath, 
    useFolderNameAsLabel: true);
IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);
Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.ImageLoadingTransformer>
    preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",
        outputColumnName: "LabelAsKey")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: assetsRelativePath,
        inputColumnName: "ImagePath"));
IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);
TrainTestData trainSplit = mlContext.Data.TrainTestSplit(
    data: preProcessedData, 
    testFraction: 0.3);
TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(
    data: trainSplit.TestSet);

IDataView trainSet = trainSplit.TrainSet;
IDataView validationSet = validationTestSplit.TrainSet;
IDataView testSet = validationTestSplit.TestSet;

//https://en.wikipedia.org/wiki/Residual_neural_network
//
ImageClassificationTrainer.Options classifierOptions = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "Image",
    LabelColumnName = "LabelAsKey",
    ValidationSet = validationSet,
    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
    MetricsCallback = (metrics) => Console.WriteLine(metrics),
    TestOnTrainSet = false,
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true
};
Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer>? trainingPipeline = 
    mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
ITransformer trainedModel = trainingPipeline.Fit(trainSet);
ClassifySingleImage(mlContext, testSet, trainedModel);
ClassifyImages(mlContext, testSet, trainedModel);

void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
{
    using PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
    ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
    ModelOutput prediction = predictionEngine.Predict(image);
    Console.WriteLine("Classifying single image");
    OutputPrediction(prediction);
}

void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
{
    IDataView predictionData = trainedModel.Transform(data);
    IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(
        data: predictionData, 
        reuseRowObject: true).Take(10);
    Console.WriteLine("Classifying multiple images");
    foreach (ModelOutput prediction in predictions)
    {
        OutputPrediction(prediction);
    }
}


void OutputPrediction(ModelOutput prediction)
{
    string imageName = Path.GetFileName(prediction.ImagePath);
    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
}

IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
    string[] files = Directory.GetFiles(folder, "*",
        searchOption: SearchOption.AllDirectories);
    for (int i = 0; i < files.Length; i++)
    {
        string file = files[i];
        if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
        { 
            continue;
        }
        string label = Path.GetFileName(file);
        DirectoryInfo? parentDirectory = Directory.GetParent(file);
        if (parentDirectory is not null)
        {
            if (useFolderNameAsLabel)
            {
                label = parentDirectory.Name;
            }
            else
            {
                for (int index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label[..index];
                        break;
                    }
                }
            }
        }
        yield return new ImageData()
        {
            ImagePath = file,
            Label = label
        };
    }
}

class ImageData
{
    public string ImagePath { get; set; }

    public string Label { get; set; }
}

class ModelInput
{
    public byte[] Image { get; set; }

    public uint LabelAsKey { get; set; }

    public string ImagePath { get; set; }

    public string Label { get; set; }
}

class ModelOutput
{
    public string ImagePath { get; set; }

    public string Label { get; set; }

    public string PredictedLabel { get; set; }
}