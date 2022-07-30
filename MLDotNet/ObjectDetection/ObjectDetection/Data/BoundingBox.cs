using System.Drawing;

namespace ObjectDetection.Data;

public record BoundingBox
{
    public BoundingBoxDimensions Dimensions { get; init; }

    public string Label { get; init; }

    public double Confidence { get; init; }

    public Lazy<RectangleF> Rect => new(() =>
        new RectangleF((float)Dimensions.X, (float)Dimensions.Y, (float)Dimensions.Width, (float)Dimensions.Height));

    public Color BoxColor { get; init; }
}