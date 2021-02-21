using Microsoft.ML;
using Microsoft.ML.Data;
using MulticlassClassification_Iris.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MulticlassClassification_Iris
{
    public static class Program
    {
        private static readonly string BaseDatasetsRelativePath = "../../../"; //relative to the assembly
        private static readonly string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/iris-train.txt";
        private static readonly string TestDataRelativePath = $"{BaseDatasetsRelativePath}/iris-test.txt";
        private static readonly string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static readonly string TestDataPath = GetAbsolutePath(TestDataRelativePath);

        private static readonly string BaseModelsRelativePath = "../../../TrainedModel";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/IrisClassificationModel";
        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        private static void Main()
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);
            FullPipeline(mlContext, TestSomePredictions);
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        private static void FullPipeline(MLContext mlContext, Action<MLContext> testMethod = null)
        {
            // Data
            // STEP 1: Common data loading configuration
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            IDataView testDataView = mlContext.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);
            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(IrisData.Label))
                .Append(mlContext.Transforms.Concatenate("Features",
                    nameof(IrisData.SepalLength),
                    nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLength),
                    nameof(IrisData.PetalWidth)))
                .AppendCacheCheckpoint(mlContext);
            // Use in-memory cache for small/medium datasets to lower training time. 
            // Do NOT use it (remove .AppendCacheCheckpoint()) when ha

            // Trainer
            // STEP 3: Set the training algorithm, then append the trainer to the pipeline  
            // Stochastic Dual Coordinate Ascent
            //https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/main-3.pdf#:~:text=Stochastic%20Dual%20Coordinate%20Ascent%20(SDCA)%20has%20recently%20emerged,random-%20coordinate%20updates%20to%20maximize%20the%20dual%20objective.
            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
                .Append(mlContext.Transforms.Conversion
                .MapKeyToValue(outputColumnName: nameof(IrisData.Label), inputColumnName: "KeyColumn"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Training
            // STEP 4: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);

            // Evaluate
            // STEP 5: Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");

            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {trainer} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("The model is saved to {0}", ModelPath);

            //Test
            testMethod?.Invoke(mlContext);
        }

        private static void TestSomePredictions(MLContext mlContext)
        {
            //Test Classification Predictions with some hard-coded samples
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
            // Create prediction engine related to the loaded trained model
            PredictionEngine<IrisData, IrisPrediction> predEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(trainedModel);

            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            float[] labelsArray = keys.DenseValues().ToArray();

            Dictionary<float, string> IrisFlowers = new Dictionary<float, string>
            {
                { 0, "Setosa" },
                { 1, "versicolor" },
                { 2, "virginica" }
            };

            Console.WriteLine("=====Predicting using model====");
            //Score sample 1
            var resultprediction1 = predEngine.Predict(SampleIrisData.Iris1);

            Console.WriteLine($"Actual: setosa.     Predicted label and score:  {IrisFlowers[labelsArray[0]]}: {resultprediction1.Score[0]:0.####}");
            Console.WriteLine($"                                                {IrisFlowers[labelsArray[1]]}: {resultprediction1.Score[1]:0.####}");
            Console.WriteLine($"                                                {IrisFlowers[labelsArray[2]]}: {resultprediction1.Score[2]:0.####}");
            Console.WriteLine();

            //Score sample 2
            var resultprediction2 = predEngine.Predict(SampleIrisData.Iris2);

            Console.WriteLine($"Actual: Virginica.   Predicted label and score:  {IrisFlowers[labelsArray[0]]}: {resultprediction2.Score[0]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[1]]}: {resultprediction2.Score[1]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[2]]}: {resultprediction2.Score[2]:0.####}");
            Console.WriteLine();

            //Score sample 3
            var resultprediction3 = predEngine.Predict(SampleIrisData.Iris3);

            Console.WriteLine($"Actual: Versicolor.   Predicted label and score: {IrisFlowers[labelsArray[0]]}: {resultprediction3.Score[0]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[1]]}: {resultprediction3.Score[1]:0.####}");
            Console.WriteLine($"                                                 {IrisFlowers[labelsArray[2]]}: {resultprediction3.Score[2]:0.####}");
            Console.WriteLine();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            return Path.Combine(assemblyFolderPath, relativePath);
        }
    }
}
