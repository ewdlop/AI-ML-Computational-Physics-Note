using Microsoft.ML;

namespace NLPRedo;

public class Trainer
{
    private readonly MLContext _mlContext;

    public Trainer(int seed = 11)
    {
        _mlContext = new MLContext(seed);
    }

    public ITransformer BuidAndTrain(string bertModelPath, bool useGpu)
    {
        Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator pipeline = _mlContext.Transforms
                        .ApplyOnnxModel(modelFile: bertModelPath,
                                        outputColumnNames: new[] { "unstack:1",
                                                                   "unstack:0",
                                                                   "unique_ids:0" },
                                        inputColumnNames: new[] {"unique_ids_raw_output___9:0",
                                                                  "segment_ids:0",
                                                                  "input_mask:0",
                                                                  "input_ids:0" },
                                        gpuDeviceId: useGpu ? 0 : null);

        return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<BertInput>()));
    }
}