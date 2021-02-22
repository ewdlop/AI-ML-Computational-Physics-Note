using System;
using System.Security.Cryptography;

namespace BasicNeuralNetwork
{
    static class RNGCryptoService
    {
        public static double Generate()
        {
            using(var p = new RNGCryptoServiceProvider())
            {
                var r = new Random(p.GetHashCode());
                return r.NextDouble();
            }
        }
    }
}
