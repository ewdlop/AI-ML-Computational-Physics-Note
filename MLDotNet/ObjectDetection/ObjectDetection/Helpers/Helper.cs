namespace ObjectDetection.Helpers;

public static class MathHelper
{
    public static double Sigmoid(double value)
    {
        double k =Math.Exp(value);
        return k / (1.0f + k);
    }

    public  static double[] Softmax(double[] values)
    {
        double maxVal = values.Max();
        IEnumerable<double> exp = values.Select(v => Math.Exp(v - maxVal));
        double sumExp = exp.Sum();

        return exp.Select(v => (v / sumExp)).ToArray();
    }

}