using OpenCvSharp;
using SkiaSharp;

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