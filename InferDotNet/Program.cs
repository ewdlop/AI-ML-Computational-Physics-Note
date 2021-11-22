﻿using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;

public static class MathHelper
{
    public static void FairTwoCoin(double chance)
    {
        //random variable
        Variable<bool> firstCoin = Variable.Bernoulli(chance);
        Variable<bool> secondCoin = Variable.Bernoulli(chance);
        Variable<bool> bothHeads = firstCoin & secondCoin;

        InferenceEngine engine = new();
        Console.WriteLine($"Probability both coins are heads:{engine.Infer(bothHeads)}");

        bothHeads.ObservedValue = false;
        Console.WriteLine("Probability distribution over firstCoin: " + engine.Infer(firstCoin));
    }

    public static void Gaussian(double mean, double variance)
    {
        Variable<double> threshold = Variable.New<double>().Named("threshold");
        Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
        Variable.ConstrainTrue(x > threshold);
        InferenceEngine engine = new InferenceEngine();
        if (engine.Algorithm is ExpectationPropagation)
        {
            for (double thresh = 0; thresh <= 1; thresh += 0.1)
            {
                threshold.ObservedValue = thresh;
                Console.WriteLine("Dist over x given thresh of " + thresh + "=" + engine.Infer(x));
            }
        }
        else
        {
            Console.WriteLine("This example only runs with Expectation Propagation");
        }
    }

    public static void LargeSampleing()
    {
        // Sample data from standard Gaussian
        double[] data = new double[100];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = Rand.Normal(0, 1);
        }

        // Create mean and precision random variables
        Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
        Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");

        Microsoft.ML.Probabilistic.Models.Range dataRange = new Microsoft.ML.Probabilistic.Models.Range(data.Length).Named("n");
        //for (int i = 0; i < data.Length; i++)  
        //{
        //    Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision);
        //    x.ObservedValue = data[i];
        //}
        VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");
        x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);
        x.ObservedValue = data;

        InferenceEngine engine = new InferenceEngine();
        //engine.ShowFactorGraph = true;

        // Retrieve the posterior distributions
        Console.WriteLine("mean=" + engine.Infer(mean));
        Console.WriteLine("prec=" + engine.Infer(precision));
    }

    public static void Main(string[] args)
    {
        //FairTwoCoin(0.5);
        //Gaussian(0, 1);
        LargeSampleing();
    }
}