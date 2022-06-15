using Microsoft.ML;
using NLP.BERT.DataModel;

namespace NLP.BERT.Trainers;

public class Trainer
{
    private readonly MLContext _mlContext;

    public Trainer()
    {
        _mlContext = new MLContext(11);
    }

    public ITransformer BindAndTrains(string bertModelPath, bool useGPU = false)
    {
        Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator pipeline = _mlContext.Transforms.ApplyOnnxModel(
            modelFile: bertModelPath,
            inputColumnNames: new[] { 
                "unique_ids_raw_output___9:0", 
                "segment_ids:0", 
                "input_mask:0", 
                "input_ids:0" 
            },
            outputColumnNames: new[]
            {
                "unstack:1", 
                "unstack:0", 
                "unique_ids:0"
            },
            gpuDeviceId: useGPU ? 0 : null);

        return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(Enumerable.Empty<BertInput>()));
    }
}