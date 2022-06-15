using Microsoft.ML;
using NLP.BERT.DataModel;

namespace NLP.BERT.Predictors;

public class Predictor
{
    private MLContext _mLContext;
    private PredictionEngine<BertInput, BertPredictions> _predictionEngine;

    public Predictor(ITransformer transformer)
    {
        _mLContext = new MLContext();
        _predictionEngine = _mLContext.Model.CreatePredictionEngine<BertInput, BertPredictions>(transformer);
    }
    public BertPredictions BertPredictions(BertInput input) 
        => _predictionEngine.Predict(input);
    
}
