namespace ObjectDetection.Data;

public record struct CellDimensions
{
    public double X { get; init; }
    public double Y { get; init; }
    public double Height { get; init; }
    public double Width { get; init; }
}
