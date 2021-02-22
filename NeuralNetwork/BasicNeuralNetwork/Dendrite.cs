namespace BasicNeuralNetwork
{
    public class Dendrite
    {
        public double Weight { get; set; }

        public Dendrite()
        {
            Weight = RNGCryptoService.Generate();
        }
    }
}
