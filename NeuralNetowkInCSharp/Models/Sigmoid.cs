using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetowkInCSharp.Models
{
    public static class Sigmoid
    {
        public static double Output(double x)
        {
            return x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double Derivative(double s)
        {
            return s * (1 - s);
        }
    }
}
