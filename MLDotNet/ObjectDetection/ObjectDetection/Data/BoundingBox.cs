using System.Drawing;

namespace ObjectDetection.Data;

public class BoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; }

    public string Label { get; set; }

    public double Confidence { get; set; }

    public Lazy<RectangleF> Rect => new(() =>
        new RectangleF((float)Dimensions.X, (float)Dimensions.Y, (float)Dimensions.Width, (float)Dimensions.Height));

    public Color BoxColor { get; set; }
}