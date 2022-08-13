using Microsoft.ML;

namespace NLPRedo;

public class Predictor
{
    private readonly MLContext _mLContext;
    private PredictionEngine<BertInput, BertPredictions> _predictionEngine;

    public Predictor(ITransformer trainedModel)
    {
        _mLContext = new MLContext();
        _predictionEngine = _mLContext.Model.CreatePredictionEngine<BertInput, BertPredictions>(trainedModel);
    }

    public BertPredictions Predict(BertInput encodedInput) => _predictionEngine.Predict(encodedInput);
}