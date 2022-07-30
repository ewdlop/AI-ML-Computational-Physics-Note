using System.Drawing;
using System.Drawing.Drawing2D;
using Microsoft.ML;
using ObjectDetection.Data;
using ObjectDetection.Scorers;
using ObjectDetection.Parser;

FileInfo root = new FileInfo(typeof(Program).Assembly.Location);
string assetsRelativePath = Path.Combine(root.Directory.FullName, "Assets");
string modelFilePath = Path.Combine(assetsRelativePath, "Model", "tinyyolov2-8.onnx");
string imagesFolder = Path.Combine(assetsRelativePath, "Images");
string outputFolder = Path.Combine(assetsRelativePath, "Images", "Output");

MLContext mlContext = new MLContext();
try
{
    ImageNetData[] images = ReadFromFile(imagesFolder);
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);
    // Create instance of model scorer
    OnnxModelScorer modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

    // Use model to score data
    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

    IList<BoundingBox>[] boundingBoxes = probabilities
            .Select(probability => OutputParser.ParseOutputs(probability))
            .Select(boxes => OutputParser.FilterBoundingBoxes(boxes, 5, .5F))
            .ToArray();

    for (var i = 0; i < images.Length; i++)
    {
        string imageFileName = images[i].Label;
        IList<BoundingBox> detectedObjects = boundingBoxes[i];
        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);
        LogDetectedObjects(imageFileName, detectedObjects);
        Console.WriteLine("========= End of Process..Hit any Key ========");
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

void DrawBoundingBox(
    string inputImageLocation,
    string outputImageLocation,
    string imageName,
    IList<BoundingBox> filteredBoundingBoxes)
{
    Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    for (int i = 0; i < filteredBoundingBoxes.Count; i++)
    {
        BoundingBox box = filteredBoundingBoxes[i];
        var x = (uint)Math.Max(box.Dimensions.X, 0);
        var y = (uint)Math.Max(box.Dimensions.Y, 0);
        var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);
        x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
        y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
        width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
        height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;
        string text = $"{box.Label} ({box.Confidence * 100:0}%)";
        
        using Graphics thumbnailGraphic = Graphics.FromImage(image);
        thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
        thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
        thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;
        
        Font drawFont = new("Arial", 12, FontStyle.Bold);
        SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
        SolidBrush fontBrush = new(Color.Black);
        Point atPoint = new((int)x, (int)y - (int)size.Height - 1);
        // Define BoundingBox options
        Pen pen = new(box.BoxColor, 3.2f);
        SolidBrush colorBrush = new(box.BoxColor);
        
        thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
        thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);
        thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
        
        if (!Directory.Exists(outputImageLocation))
        {
            Directory.CreateDirectory(outputImageLocation);
        }
        image.Save(Path.Combine(outputImageLocation, imageName));
    }

}


void LogDetectedObjects(string imageName, IList<BoundingBox> boundingBoxes)
{
    Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

    for (int i = 0; i < boundingBoxes.Count; i++)
    {
        BoundingBox box = boundingBoxes[i];
        Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
    }
    Console.WriteLine();
}

ImageNetData[] ReadFromFile(string imageFolder) => Directory
        .GetFiles(imageFolder)
        .Select(filePath => new ImageNetData
        {
            ImagePath = filePath,
            Label = Path.GetFileName(filePath)
        }).ToArray();
