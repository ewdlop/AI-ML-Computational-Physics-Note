using Microsoft.ML.Data;

namespace ObjectDetection.Data;

public class ImageNetPrediction
{
    [ColumnName("grid")]
    public float[] PredictedLabels;
}