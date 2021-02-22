using System;
using System.Collections.Generic;

namespace BasicNeuralNetwork
{
    public class Neuron
    {
        public List<Dendrite> Dendrites { get; set; }
        public double Bias { get; set; }
        public double Delta { get; set; }
        public double Value { get; set; }

        public int DendriteCount => Dendrites.Count;

        public Neuron()
        {
            Random n = new Random(Environment.TickCount);
            Bias = n.NextDouble();

            Dendrites = new List<Dendrite>();
        }
    }
}
