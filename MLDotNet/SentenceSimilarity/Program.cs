using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using MathNet.Numerics.Statistics;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Data.TextLoader;

// Initialize MLContext
MLContext? ctx = new MLContext();


// (Optional) Use GPU
ctx.GpuDeviceId = 0;
ctx.FallbackToCpu = true;


// Log training output
ctx.Log += (o, e) => {
    if (e.Source.Contains("NasBertTrainer"))
        Console.WriteLine(e.Message);
};


// Load data into IDataView
Column[]? columns = new[]
{
    new TextLoader.Column("search_term",DataKind.String,3),
    new TextLoader.Column("relevance",DataKind.Single,4),
    new TextLoader.Column("product_description",DataKind.String,5)
};

Options? loaderOptions = new TextLoader.Options()
{
    Columns = columns,
    HasHeader = true,
    Separators = new[] { ',' },
    MaxRows = 1000 // Dataset has 75k rows. Only load 1k for quicker training
};

Console.WriteLine("Loading data...");
var dataPath = Path.GetFullPath("train.csv");
var textLoader = ctx.Data.CreateTextLoader(loaderOptions);
var data = textLoader.Load(dataPath);
Console.WriteLine("Data loaded.");

// Split data into 80% training, 20% testing
Console.WriteLine("Splitting data...");
var dataSplit = ctx.Data.TrainTestSplit(data, testFraction: 0.2);

// Create data processing pipeline
Console.WriteLine("Creating data processing pipeline...");
// Define pipeline
// NAS-BERT is a pretrained model, so we don't need to train it
var pipeline =
    ctx.Transforms.ReplaceMissingValues("relevance", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
    .Append(ctx.Regression.Trainers.SentenceSimilarity( 
        labelColumnName: "relevance",
        sentence1ColumnName: "search_term",
        sentence2ColumnName: "product_description"));

// Fit pipeline to training data
Console.WriteLine("Fitting pipeline to training data...");
// Train the model
var model = pipeline.Fit(dataSplit.TrainSet);

// Evaluate model on test data
Console.WriteLine("Evaluating model on test data...");
// Use the model to make predictions on the test dataset
var predictions = model.Transform(dataSplit.TestSet);

// Evaluate the model
Evaluate(predictions, "relevance", "Score");

// Save the model
Console.WriteLine("Saving model...");
ctx.Model.Save(model, data.Schema, "model.zip");

Console.WriteLine("Done");

static void Evaluate(IDataView predictions, string actualColumnName, string predictedColumnName)
{
    IEnumerable<double> actual =
        predictions.GetColumn<float>(actualColumnName)
            .Select(x => (double)x);
    IEnumerable<double> predicted =
        predictions.GetColumn<float>(predictedColumnName)
            .Select(x => (double)x);
    double corr = Correlation.Pearson(actual, predicted);
    Console.WriteLine($"Pearson Correlation: {corr}");
}