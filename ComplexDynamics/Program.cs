using SkiaSharp;
using System.IO;
const int Width = 1920;
const int Height = 1080;
const int MaxIterations = 1000;
const float HueFactor = 360.0f / MaxIterations; // Assuming MaxIterations is the total number of iterations
const float HueOffset = 0.625f; // The original value was (5.0 / 8.0)
const float Saturation = 1.0f; // Fully saturated
const float ValueBase = 1.0f;  // This represents full brightness

SKBitmap bitmap = GenerateJuliaSet(Width, Height, MaxIterations);
SaveBitmap(bitmap, "julia.png");
static SKBitmap GenerateJuliaSet(int width, int height, int maxIterations)
{
    SKBitmap bitmap = new SKBitmap(width, height);
    SKColor[] colors = GenerateColors(maxIterations);

    double c_re = -0.1;
    double c_im = 0.56;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int iteration = CalculateIteration(col, row, width, height, c_re, c_im, maxIterations);
            bitmap.SetPixel(col, row, iteration < maxIterations ? colors[iteration] : SKColors.White);
        }
    }
    return bitmap;
}

static SKColor[] GenerateColors(int maxIterations)
{
    SKColor[] colors = new SKColor[maxIterations];

    for (int i = 0; i < maxIterations; i++)
    {
        // Cycle through the hue range based on iteration
        float hue = (i * HueFactor + HueOffset * 360) % 360;

        // For a vibrant gradient, keep saturation high
        float saturation = Saturation;

        // Keeping value high ensures colors are bright
        float value = ValueBase;

        colors[i] = SKColor.FromHsv(hue, saturation, value);
    }

    return colors;
}


static int CalculateIteration(int col, int row, int width, int height, double c_re, double c_im, int maxIterations)
{
    double x = (col - width / 2.0) * (3.5 / width);  // Adjust the 3.5 as needed for zoom
    double y = (row - height / 2.0) * (3.5 / height); // Adjust the 3.5 as needed for zoom

    int iteration = 0;
    while (x * x + y * y < 4 && iteration < maxIterations)
    {
        double x_new = x * x - y * y + c_re;
        y = 2 * x * y + c_im;
        x = x_new;
        iteration++;
    }
    return iteration;
}


static void SaveBitmap(SKBitmap bitmap, string filename)
{
    SKImage image = SKImage.FromBitmap(bitmap);
    using SKData data = image.Encode(SKEncodedImageFormat.Png, 100);
    using FileStream stream = File.OpenWrite(filename);
    data.SaveTo(stream);
}