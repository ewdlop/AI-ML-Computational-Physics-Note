﻿using Microsoft.ML.Data;

namespace ObjectDetection.Data;

public class ImageNetData
{
    [LoadColumn(0)]
    public string ImagePath { get; set; }

    [LoadColumn(1)]
    public string Label { get; set; }
}
