using Microsoft.ML.Data;

namespace NLP.BERT.DataModel;

public class BertPredictions
{
    [VectorType(1, 256)]
    [ColumnName("unstack:1")]
    public float[]? EndLogistics { get; set; }

    [VectorType(1, 256)]
    [ColumnName("unstack:0")]
    public float[]? StartLogistics { get; set; }

    [VectorType(1)]
    [ColumnName("unique_ids:0")]
    public long[]? UniqueIds { get; set; }
}