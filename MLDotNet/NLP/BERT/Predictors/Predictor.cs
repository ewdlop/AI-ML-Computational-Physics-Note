using NLP.BERT.DataModel;

namespace NLP.BERT.Predictors;

public class Predictor
{
    private Microsoft.ML.MLContext _mLContext;
    private Microsoft.ML.PredictionEngine<BertInput, BertPredictions> _predictionEngine;

    public Predictor(Microsoft.ML.ITransformer transformer)
    {
        _mLContext = new Microsoft.ML.MLContext();
        _predictionEngine = _mLContext.Model.CreatePredictionEngine<BertInput, BertPredictions>(transformer);
    }
    public BertPredictions Predict(BertInput encodedInput) 
        => _predictionEngine.Predict(encodedInput);

}
