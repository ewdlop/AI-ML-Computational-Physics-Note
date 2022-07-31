using Bogus;
using ImageSegmentation.Extractors;
using OpenCvSharp;
using SkiaSharp;
using System.Collections.Immutable;



//https://debuggercafe.com/image-foreground-extraction-using-opencv-contour-detection/
WaterShed();
//DectectFacesAsync();
//CropImage();


//https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
//not finisheed
//might not work with bright hair.....
void WaterShed()
{
    string assetsPath = Path.Combine(Environment.CurrentDirectory, "Assets");
    string imagesPath = Path.Combine(assetsPath, "Images");
    string imagePath = Path.Combine(imagesPath, "fauci.jpg");

    using Mat image = Cv2.ImRead(imagePath);
    using Mat gray = new Mat();
    using Mat threshold = new Mat();
    using Mat morphology = new Mat();
    Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
    Cv2.ImShow("test", gray);
    Cv2.WaitKey(0);
    Cv2.Threshold(gray, threshold, 150, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
    Cv2.ImShow("test", threshold);
    Cv2.WaitKey(0);
    Cv2.MorphologyEx(
        threshold,
        morphology,
        MorphTypes.Open,
        new Mat(3, 3, MatType.CV_8U, new byte[] { 0, 1, 0, 1, 1, 1, 0, 1, 0 }));
    Cv2.ImShow("test", morphology);
    Cv2.WaitKey(0);
}
void CropImage()
{
    string assetsPath = Path.Combine(Environment.CurrentDirectory, "Assets");
    string facexmlPath = Path.Combine(assetsPath, "haarcascade_frontalface_alt2.xml");
    string imagesPath = Path.Combine(assetsPath, "Images");
    string imagePath = Path.Combine(imagesPath, "Sparrow-PNG-Image-71269-768x622.png");
    string imageOutputName = "fauci_test.png";

    using Mat image = Cv2.ImRead(imagePath);
    using Mat image2 = Cv2.ImRead(imagePath);
    using Mat blurred = new Mat();
    using Mat gray = new Mat();
    using Mat threshold = new Mat();
    
    for (int i = 0; i < 256; i++)
    {
        try
        {
            //Cv2.GaussianBlur(image, blurred, new Size(i, i), 0);
            Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.Threshold(gray, threshold, i, 255, ThresholdTypes.Otsu);
            Point[] contour = Extractor.FindLargetsContourToArray(threshold);
            //Point[][] contours = Cv2.FindContoursAsArray(
            //    threshold, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
            Cv2.DrawContours(
                image2,
                contours: new Point[1][] { contour },
                -1,
                new Scalar(0, 0, 255),
                2,
                LineTypes.AntiAlias,
                maxLevel: 1);
            Cv2.ImWrite($"Test/{i}.jpg", image2);
        }
        catch (Exception)
        {

        }
    }

    //Cv2.ImShow(imageOutputName, image2);
    //Cv2.WaitKey(0);

}
async Task DectectFacesAsync()
{
    string assetsPath = Path.Combine(Environment.CurrentDirectory, "Assets");
    string facexmlPath = Path.Combine(assetsPath, "haarcascade_frontalface_alt2.xml");
    string imagesPath = Path.Combine(assetsPath, "Images");
    string imageName = "family.jpg";
    string imageOutputName = "family_new.png";
    using CascadeClassifier haarCascade = new CascadeClassifier(facexmlPath);

    // Load target image
    using Mat gray = new Mat(Path.Combine(imagesPath, imageName), ImreadModes.Grayscale);

    // Detect faces
    Rect[] faces = haarCascade.DetectMultiScale(
                        gray, 1.08, 2, HaarDetectionTypes.ScaleImage, new Size(30, 30));

    byte[] bytes = await File.ReadAllBytesAsync(Path.Combine(imagesPath, imageName));
    SKBitmap bitmap = SKBitmap.Decode(bytes);
    using SKCanvas canvas = new SKCanvas(bitmap);
    SKPaint sKPaint = new SKPaint()
    {
        Color = SKColors.Red,
        Style = SKPaintStyle.Stroke
    };
    for (int i = 0; i < faces.Length; i++)
    {
        Rect rect = faces[i];
        canvas.DrawRect(rect.X, rect.Y, rect.Width, rect.Height, sKPaint);
    }
    canvas.Flush();
    using SKData data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
    await File.WriteAllBytesAsync(imageOutputName, data.ToArray());
}
