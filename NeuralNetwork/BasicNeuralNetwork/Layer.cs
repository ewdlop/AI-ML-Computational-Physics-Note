using System.Collections.Generic;

namespace BasicNeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; set; }
        public int NeuronCount => Neurons.Count;

        public Layer(int numNeurons)
        {
            Neurons = new List<Neuron>(numNeurons);
        }
    }
}
