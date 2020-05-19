using System;

namespace NeuralNetowkInCSharp.Models
{
    public class Synapse
    {
        public Guid Id { get; set; }
        public Neuron InputNeuron { get; set; }
        public Neuron OutputNeuron { get; set; }
        public double Weight { get; set; }
        public double WeightDelta { get; set; }

        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
        {
            Id = Guid.NewGuid();
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Weight = Network.GetRandomNumber(-1.0,1.0);
        }
    }
}
