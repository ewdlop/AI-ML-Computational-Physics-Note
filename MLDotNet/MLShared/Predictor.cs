using Microsoft.Extensions.ObjectPool;
using Microsoft.ML;

namespace TextClassification;

public class PredictorPool<TData, TPrediction>
    where TData : class
    where TPrediction : class, new()
{
    private readonly MLContext _mLContext;
    private readonly ObjectPool<PredictionEngine<TData, TPrediction>> _predictionEnginePool;
    private readonly ITransformer _mLModel;
    private readonly int _maxObjectsRetained;

    public PredictorPool(MLContext mLContext, ITransformer mlModel, string? modelPath = null, int seed = 11, int maxObjectsRetained = -1)
    {
        _mLContext = mLContext;
        _mLModel = mlModel;
        _maxObjectsRetained = maxObjectsRetained;
        _predictionEnginePool = CreatePredictionEngineObjectPool();

    }
    private ObjectPool<PredictionEngine<TData, TPrediction>> CreatePredictionEngineObjectPool()
    {
        PooledPredictionEnginePolicy<TData, TPrediction> predEnginePolicy = new(_mLContext, _mLModel);

        DefaultObjectPool<PredictionEngine<TData, TPrediction>> pool;

        if (_maxObjectsRetained != -1)
        {
            pool = new DefaultObjectPool<PredictionEngine<TData, TPrediction>>(predEnginePolicy, _maxObjectsRetained);
        }
        else
        {
            //default maximumRetained is Environment.ProcessorCount * 2, if not explicitly provided
            pool = new DefaultObjectPool<PredictionEngine<TData, TPrediction>>(predEnginePolicy);
        }

        return pool;
    }

    public TPrediction Predict(TData encodedInput)
    {
        //Get PredictionEngine object from the Object Pool
        PredictionEngine<TData, TPrediction> predictionEngine = _predictionEnginePool.Get();

        try
        {
            //Predict
            TPrediction prediction = predictionEngine.Predict(encodedInput);
            return prediction;
        }
        finally
        {
            //Release used PredictionEngine object into the Object Pool
            _predictionEnginePool.Return(predictionEngine);
        }
    }
}
