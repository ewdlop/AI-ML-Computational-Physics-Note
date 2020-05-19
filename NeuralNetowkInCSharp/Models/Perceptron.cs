using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetowkInCSharp.Models
{
    public class Perceptron
    {
        private float[] weights;
        public Perceptron(int n)
        {
            weights = new float[n];
            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] = (float)Network.GetRandomNumber(-1f, 1f);
            }
        }
    }
}
