using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

namespace TextClassification;

public class RegressionTrainer
{
    private readonly MLContext _mlContext;
    private ITransformer _model;
    private PredictorPool<RegressionInput, RegressionOutput> _predictorPool;
    public RegressionTrainer(RegressionInput[] data)
    {
        int seed = 11;
        _mlContext = new MLContext(seed)
        {
            GpuDeviceId = 0,
            FallbackToCpu = false
        };

        IDataView dataView = _mlContext.Data.LoadFromEnumerable(data);
        SentenceSimilarityTrainer pipeline =
            _mlContext.Regression.Trainers.SentenceSimilarity(
                labelColumnName: "Similarity",
                sentence1ColumnName: "Source",
                sentence2ColumnName: "Target");
        _model = pipeline.Fit(dataView);
        _predictorPool = new PredictorPool<RegressionInput, RegressionOutput>(_mlContext, _model);
        
    }

    public RegressionOutput Predict(RegressionInput encodedInput)
    {
        return _predictorPool.Predict(encodedInput);
    }
}

public record RegressionInput
{
    [ColumnName("Source")]
    public string Source { get; init; } = string.Empty;
    [ColumnName("Target")]
    public string Target { get; init; } = string.Empty;
    [ColumnName("Similarity")]
    public float Similarity { get; init; }
}
public record RegressionOutput
{
    [ColumnName("Similarity")] 
    public float Similarity { get; init; }
}
