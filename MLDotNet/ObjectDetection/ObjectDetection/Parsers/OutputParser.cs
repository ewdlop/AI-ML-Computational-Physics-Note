using ObjectDetection.Data;
using ObjectDetection.Helpers;
using System.Drawing;

namespace ObjectDetection.Parser;

public static class OutputParser
{
    public const int ROW_COUNT = 13; //output layer row count
    public const int COL_COUNT = 13; // output layer column ocunt
    public const int CHANNEL_COUNT = (CLASS_COUNT + BOX_INFO_FEATURE_COUNT) * BOXES_PER_CELL;
    public const int BOXES_PER_CELL = 5;
    public const int BOX_INFO_FEATURE_COUNT = 5;
    public const int CLASS_COUNT = 20; //class probabilities for each of the 20 classes predicted by the model.
    public const float CELL_WIDTH = 32;
    public const float CELL_HEIGHT = 32;
    private const int CHANNEL_STRIDE = ROW_COUNT * COL_COUNT;
    private static readonly Lazy<float[]> Anchors= new(()=>new float[]
    {
        1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
    });
    private static readonly Lazy<string[]> Labels = new(() => new string[]
    {
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    });

    private static readonly Lazy<Color[]> ClassColors = new(()=>new Color[]
    {
        Color.Khaki,
        Color.Fuchsia,
        Color.Silver,
        Color.RoyalBlue,
        Color.Green,
        Color.DarkOrange,
        Color.Purple,
        Color.Gold,
        Color.Red,
        Color.Aquamarine,
        Color.Lime,
        Color.AliceBlue,
        Color.Sienna,
        Color.Orchid,
        Color.Tan,
        Color.LightPink,
        Color.Yellow,
        Color.HotPink,
        Color.OliveDrab,
        Color.SandyBrown,
        Color.DarkTurquoise
    });

    /// <summary>
    /// One dimensional Model
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="channel"></param>
    /// <returns></returns>
    private static int GetOffset(int x, int y, int channel)
    {
        return (channel * CHANNEL_STRIDE) + (y * COL_COUNT) + x;
    }

    public static BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
    {
        return new BoundingBoxDimensions
        {
            X = modelOutput[GetOffset(x, y, channel)],
            Y = modelOutput[GetOffset(x, y, channel + 1)],
            Width = modelOutput[GetOffset(x, y, channel + 2)],
            Height = modelOutput[GetOffset(x, y, channel + 3)]
        };
    }

    public static double GetConfidence(float[] modelOutput, int x, int y, int channel)
    {
        return MathHelper.Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]); //
    }

    public static CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
    {
        return new CellDimensions
        {
            X = (x + MathHelper.Sigmoid(boxDimensions.X)) * CELL_WIDTH,
            Y = (y + MathHelper.Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
            Width = Math.Exp(boxDimensions.Width) * CELL_WIDTH * Anchors.Value[box * 2],
            Height = Math.Exp(boxDimensions.Height) * CELL_HEIGHT * Anchors.Value[box * 2 + 1]
        };
    }

    public static double[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
    {
        double[] predictedClasses = new double[CLASS_COUNT];
        int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
        for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
        {
            predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
        }
        return MathHelper.Softmax(predictedClasses);
    }
    
    public static (int, double) GetTopResult(double[]? predictedClasses) => predictedClasses is null
            ? throw new ArgumentNullException(nameof(predictedClasses))
            : predictedClasses.Length == 0
            ? throw new ArgumentOutOfRangeException(nameof(predictedClasses), "predictedClasses is empty")
            : ((int, double))predictedClasses
            .Select((predictedClass, index) => (Index: index, Value: predictedClass))
            .OrderByDescending(result => result.Value)
            .First();

    public static double IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
    {
        double areaA = boundingBoxA.Width * boundingBoxA.Height;

        if (areaA <= 0)
            return 0;

        double areaB = boundingBoxB.Width * boundingBoxB.Height;

        if (areaB <= 0)
            return 0;

        double minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
        double minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
        double maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
        double maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

        double intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    public static IList<BoundingBox> ParseOutputs(float[] modelOutputs, float threshold = .3F)
    { 
        List<BoundingBox> boxes = new List<BoundingBox>();
        for (int row = 0; row < ROW_COUNT; row++)
        {
            for (int column = 0; column < COL_COUNT; column++)
            {
                for (int box = 0; box < BOXES_PER_CELL; box++)
                {
                    //0,25,50..,125
                    int channel = box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT);
                    double confidence = GetConfidence(modelOutputs, row, column, channel);
                    BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(modelOutputs, row, column, channel);
                    CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);
                    if (confidence < threshold) continue;
                    double[] predictedClasses = ExtractClasses(modelOutputs, row, column, channel);
                    if (Labels.Value.Length < BOXES_PER_CELL || ClassColors.Value.Length < BOXES_PER_CELL)
                    {
                        Console.WriteLine("Warning: Either Label or ClassColor size is less than the number of boxes per cell");
                    }
                    (int topResultIndex, double topResultScore) = GetTopResult(predictedClasses);
                    double topScore = topResultScore * confidence;
                    if (topScore < threshold) continue;
                    boxes.Add(new BoundingBox()
                    {
                        Dimensions = new BoundingBoxDimensions
                        {
                            X = mappedBoundingBox.X - mappedBoundingBox.Width / 2,
                            Y = mappedBoundingBox.Y - mappedBoundingBox.Height / 2,
                            Width = mappedBoundingBox.Width,
                            Height = mappedBoundingBox.Height,
                        },
                        Confidence = topScore,
                        Label = Labels.Value[topResultIndex % Labels.Value.Length],
                        BoxColor = ClassColors.Value[topResultIndex % Labels.Value.Length]
                    });
                }
            }
        }
        return boxes;
    }
    public static IList<BoundingBox> FilterBoundingBoxes(IList<BoundingBox> boxes, int limit, float threshold)
    {
        int activeCount = boxes.Count;
        bool[]? isActiveBoxes = new bool[boxes.Count];

        for (int i = 0; i < isActiveBoxes.Length; i++)
        {
            isActiveBoxes[i] = true;
        }

        (BoundingBox Box, int Index)[] sortedBoxes = boxes.Select((b, i) => (Box : b, Index : i ))
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToArray();

        List<BoundingBox> results = new List<BoundingBox>();

        for (int i = 0; i < boxes.Count; i++)
        {
            if (isActiveBoxes[i])
            {
                var boxA = sortedBoxes[i].Box;
                results.Add(boxA);

                if (results.Count >= limit) break;

                for (var j = i + 1; j < boxes.Count; j++)
                {
                    if (isActiveBoxes[j])
                    {
                        BoundingBox boxB = sortedBoxes[j].Box;

                        if (IntersectionOverUnion(boxA.Rect.Value, boxB.Rect.Value) > threshold)
                        {
                            isActiveBoxes[j] = false;
                            if (activeCount-- <= 0) break;
                        }
                    }
                }

                if (activeCount <= 0) break;
            }
        }
        return results;
    }
}