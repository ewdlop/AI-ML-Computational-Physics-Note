using OpenCvSharp;

namespace ImageSegmentation.Extractors;

/// <summary>
/// https://debuggercafe.com/image-foreground-extraction-using-opencv-contour-detection/
/// </summary>
public static class Extractor
{
    public static Mat<Point>? FindLargetsContour(Mat image)
    {
        Mat<Point>[] contours = Cv2.FindContoursAsMat(
            image, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

        return contours.MaxBy(c => c.ContourArea());
    }

    public static Point[] FindLargetsContourToArray(Mat image)
    {
        Mat<Point>[] contours = Cv2.FindContoursAsMat(
            image, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);

        return contours.MaxBy(c => c.ContourArea()).ToArray();
    }

    public static void ApplyNewBackground(string backImagePath, string outputImagePath, InputArray mask3d, InputArray foreground)
    {
        //type conversion issue...
        //List<double> x = new List<double>();
        //OutputArray outputArray = OutputArray.Create(x);
        //Cv2.Multiply(mask3d, foreground, outputArray);
        //Mat background = Cv2.ImRead(backImagePath);
        //Cv2.Resize(background,background, new Size(foreground.Rows(), foreground.Cols()) );
        //Cv2.Add(foreground, background, outputArray);
        //Cv2.ImWrite(outputImagePath, Mat.FromArray<double> (outputArray));
    }
}
