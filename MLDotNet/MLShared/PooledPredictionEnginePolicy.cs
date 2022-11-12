using Microsoft.Extensions.ObjectPool;
using Microsoft.ML;

namespace TextClassification;

public class PooledPredictionEnginePolicy<TData, TPrediction>
    : IPooledObjectPolicy<PredictionEngine<TData, TPrediction>>
               where TData : class
               where TPrediction : class, new()
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    public PooledPredictionEnginePolicy(MLContext mlContext, ITransformer model)
    {
        _mlContext = mlContext;
        _model = model;
    }

    public PredictionEngine<TData, TPrediction> Create()
    {
        return _mlContext.Model.CreatePredictionEngine<TData, TPrediction>(_model); ;
    }

    public bool Return(PredictionEngine<TData, TPrediction> obj)
    {
        return obj != null;
    }
}