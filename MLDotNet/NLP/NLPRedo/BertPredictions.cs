﻿using Microsoft.ML.Data;

namespace NLPRedo;

public class BertPredictions
{
    [VectorType(1, 256)]
    [ColumnName("unstack:1")]
    public float[]? EndLogits { get; init; }

    [VectorType(1, 256)]
    [ColumnName("unstack:0")]
    public float[]? StartLogits { get; init; }

    [VectorType(1)]
    [ColumnName("unique_ids:0")]
    public long[]? UniqueIds { get; init; }
}