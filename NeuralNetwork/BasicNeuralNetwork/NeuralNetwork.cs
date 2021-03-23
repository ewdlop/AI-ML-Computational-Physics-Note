using System;
using System.Collections.Generic;

namespace BasicNeuralNetwork
{
    interface INeuralNetwork
    {
        double[] Run(in List<double> input);
        bool Train(List<double> input, List<double> target);
    }

    public class NeuralNetwork : INeuralNetwork
    {
        public List<Layer> Layers { get; set; }
        public double LearningRate { get; set; }
        public int LayerCount => Layers.Count;

        public NeuralNetwork(in double learningRate, int[] layers)
        {
            if (layers.Length < 2) return;

            LearningRate = learningRate;//[0,1]
            Layers = new List<Layer>();

            for (int l = 0; l < layers.Length; l++)
            {
                Layer layer = new Layer(layers[l]);
                Layers.Add(layer);

                for (int n = 0; n < layers[l]; n++)
                {
                    layer.Neurons.Add(new Neuron());
                }

                layer.Neurons.ForEach((nn) => {
                    if (l != 0)
                    {
                        for (int d = 0; d < layers[l - 1]; d++)
                        {
                            nn.Dendrites.Add(new Dendrite());
                        }
                    }
                    else
                    {
                        nn.Bias = 0;
                    }
                });
            }
        }

        private static double Sigmoid(in double x) //a choice of Activation Function
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double[] Run(in List<double> input)
        {
            if (input.Count != Layers[0].NeuronCount) return null;

            for (int l = 0; l < Layers.Count; l++)
            {
                Layer layer = Layers[l];

                for (int n = 0; n < layer.Neurons.Count; n++)
                {
                    var neuron = layer.Neurons[n];

                    if (l == 0)
                    {
                        neuron.Value = input[n];
                    }
                    else
                    {
                        neuron.Value = 0;
                        for (int np = 0; np < Layers[l - 1].Neurons.Count; np++)
                        {
                            neuron.Value += Layers[l - 1].Neurons[np].Value * neuron.Dendrites[np].Weight;
                        }
                        //Actviation function
                        neuron.Value = Sigmoid(neuron.Value + neuron.Bias);
                    }
                }
            }

            Layer last = Layers[Layers.Count - 1];
            int numOutput = last.Neurons.Count;
            double[] output = new double[numOutput];
            for (int i = 0; i < last.Neurons.Count; i++)
            {
                output[i] = last.Neurons[i].Value;
            }

            return output;
        }

        //https://en.wikipedia.org/wiki/Backpropagation
        public bool Train(List<double> input, List<double> target)
        {
            try
            {
                if ((input.Count != Layers[0].Neurons.Count) || (target.Count != Layers[^1].Neurons.Count)) return false;

                Run(input);

                //with Sigmoid
                //output layer
                for (int j = 0; j < Layers[^1].Neurons.Count; j++)
                {
                    //o_j
                    var neuron = Layers[^1].Neurons[j];
                    //delta_j
                    //derivative of sigmoid(x) respect t x is x(1-x)
                    neuron.Delta = neuron.Value * (1 - neuron.Value) * (neuron.Value - target[j]);
                }

                //inner layers(all layers, excluding input and output layers)
                for (int k = Layers.Count - 2; k > 0; k--)//going back for "back"propagatin
                {
                    for (int j = 0; j < Layers[k].Neurons.Count; j++)
                    {
                        var neuron = Layers[k].Neurons[j];

                        //delta_j
                        for (int l = 0; l < Layers[k + 1].Neurons.Count; l++)
                        {
                            //derivative of sigmoid(x) respect t x is x(1-x)
                            neuron.Delta += neuron.Value *
                              (1 - neuron.Value) *
                              Layers[k + 1].Neurons[l].Dendrites[j].Weight *
                              Layers[k + 1].Neurons[l].Delta;
                        }
                    }
                }

                for (int k = Layers.Count - 1; k > 0; k--)
                {
                    for (int j = 0; j < Layers[k].Neurons.Count; j++)
                    {
                        var n = Layers[k].Neurons[j];
                        n.Bias += LearningRate * n.Delta;

                        for (int i = 0; i < n.DendriteCount; i++)
                        {
                            //w_ij
                            //updating weight
                            n.Dendrites[i].Weight += LearningRate * Layers[k - 1].Neurons[i].Value * n.Delta;
                        }
                    }
                }
                return true;
            }
            catch(Exception)
            {
                throw;
            }
        }
    }
}
