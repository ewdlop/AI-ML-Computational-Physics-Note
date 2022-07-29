using System.Drawing;
using System.Drawing.Drawing2D;
using ObjectDetection;
using Microsoft.ML;
using ObjectDetection.Data;

string assetsRelativePath = "./Assets";
string modelFilePath = Path.Combine(assetsRelativePath, "Model", "TinyYolo2_model.onnx");
string imagesFolder = Path.Combine(assetsRelativePath, "Images");
string outputFolder = Path.Combine(assetsRelativePath, "Images", "Output");

MLContext mlContext = new MLContext();

static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
{
    return Directory
        .GetFiles(imageFolder)
        .Select(filePath => new ImageNetData 
        { 
            ImagePath = filePath, 
            Label = Path.GetFileName(filePath) 
        });
}
