using Microsoft.ML;
using Microsoft.ML.Data;
using ObjectDetection.Data;

namespace ObjectDetection.Scorers;

public class OnnxModelScorer
{
    public struct ImageNetSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }
    public struct TinyModelSettings
    {
        // for checking Tiny yolo2 Model input and  output  parameter names,
        //you can use tools like Netron, 
        // which is installed by Visual Studio AI Tools

        // input tensor name
        public const string ModelInput = "image";

        // output tensor name
        public const string ModelOutput = "grid";
    }
    private readonly string _imagesFolder;
    private readonly string _modelLocation;
    private readonly MLContext _mlContext;

    private IList<BoundingBox> _boundingBoxes = new List<BoundingBox>();
    public OnnxModelScorer(string imagesFolder,
                           string modelLocation,
                           MLContext mlContext)
    {
        _imagesFolder = imagesFolder;
        _modelLocation = modelLocation;
        _mlContext = mlContext;
    }

    private ITransformer LoadModel(string modelLocation)
    {
        Console.WriteLine("Read model");
        Console.WriteLine($"Model location: {modelLocation}");
        Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

        IDataView data = _mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

        EstimatorChain<Microsoft.ML.Transforms.Onnx.OnnxTransformer> pipeline = _mlContext.Transforms.LoadImages(
            outputColumnName: "image",
            imageFolder: string.Empty,
            inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages(
                    outputColumnName: "image",
                    imageWidth: ImageNetSettings.imageWidth,
                    imageHeight: ImageNetSettings.imageHeight,
                    inputColumnName: "image"))
                .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
                .Append(_mlContext.Transforms.ApplyOnnxModel(
                    modelFile: modelLocation,
                    outputColumnNames: new[] { TinyModelSettings.ModelOutput },
                    inputColumnNames: new[] { TinyModelSettings.ModelInput }));
        
        TransformerChain<Microsoft.ML.Transforms.Onnx.OnnxTransformer> model = pipeline.Fit(data);

        return model;
    }

    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        Console.WriteLine($"Images location: {_imagesFolder}");
        Console.WriteLine("");
        Console.WriteLine("=====Identify the objects in the images=====");
        Console.WriteLine("");

        IDataView scoredData = model.Transform(testData);
        IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyModelSettings.ModelOutput);

        return probabilities;
    }

    public IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(_modelLocation);

        return PredictDataUsingModel(data, model);
    }
}
