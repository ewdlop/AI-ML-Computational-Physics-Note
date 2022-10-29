using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Models;
using Range = Microsoft.ML.Probabilistic.Models.Range;

namespace InferDotNet;

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

        //Going backwards
        //Suppose it was not not observed
        bothHeads.ObservedValue = false;
        Console.WriteLine($"Probability distribution over firstCoin: {engine.Infer(firstCoin)}");
    }

    public static void TruncatedGaussian(double mean, double variance)
    {
        //Conditional Random Variable
        Variable<double> threshold = Variable.New<double>().Named("threshold");
        Variable<double> x = Variable.GaussianFromMeanAndVariance(0, 1).Named("x");
        Variable.ConstrainTrue(x > threshold);
        InferenceEngine engine = new InferenceEngine();
        if (engine.Algorithm is ExpectationPropagation)
        {
            for (double thresh = 0; thresh <= 1; thresh += 0.1)
            {
                threshold.ObservedValue = thresh;
                //moment-matched Gaussian distribution.
                Console.WriteLine($"Dist over x given thresh of {thresh}= {engine.Infer(x)}");
            }
        }
        else
        {
            Console.WriteLine("This example only runs with Expectation Propagation");
        }
    }

    public static void LargeSampling()
    {
        // Sample 100 data from standard Gaussian, observed value
        double[] data = new double[100];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = Rand.Normal(0, 1);
        }

        // Create mean and precision random variables
        Variable<double> mean = Variable.GaussianFromMeanAndVariance(0, 100).Named("mean");
        Variable<double> precision = Variable.GammaFromShapeAndScale(1, 1).Named("precision");


        //this is too slow
        //for (int i = 0; i < data.Length; i++)  
        //{
        //    Variable<double> x = Variable.GaussianFromMeanAndPrecision(mean, precision);
        //    x.ObservedValue = data[i];
        //}

        Range dataRange = new Range(data.Length).Named("n");
        VariableArray<double> x = Variable.Array<double>(dataRange).Named("x");

        //To refer to each element of the array, we index the array by the range
        x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange);
        x.ObservedValue = data;

        InferenceEngine engine = new InferenceEngine();
        //engine.ShowFactorGraph = true;

        // Retrieve the posterior distributions
        Console.WriteLine($"mean= {engine.Infer(mean)}");
        Console.WriteLine($"precision= {engine.Infer(precision)}");
    }

    public static void BayesPointMachineMotive()
    {
        double[] incomes = { 63, 16, 28, 55, 22, 20 };
        double[] ages = { 38, 23, 40, 27, 18, 40 };
        bool[] willBuy = { true, false, true, true, false, false };

        // Create x vector, augmented by 1
        Vector[] xdata = new Vector[incomes.Length];
        for (int i = 0; i < xdata.Length; i++)
        {
            xdata[i] = Vector.FromArray(incomes[i], ages[i], 1);
        }
        VariableArray<Vector> x = Variable.Observed(xdata);
        // Create target y  
        VariableArray<bool> y = Variable.Observed(willBuy, x.Range);
    }

    public static void TestingBayesPointMachine()
    {
        Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3)));
        InferenceEngine engine = new InferenceEngine(new ExpectationPropagation());
        VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
        double[] incomesTest = { 58, 18, 22 };
        double[] agesTest = { 36, 24, 37 };
        VariableArray<bool> ytest = Variable.Array<bool>(new Range(agesTest.Length));
        BayesPointMachine(incomesTest, agesTest, Variable.Random(wPosterior), ytest);
        Console.WriteLine($"output=\n{engine.Infer(ytest)}");
    }

    public static void BayesPointMachine(double[] incomes, double[] ages, Variable<Vector> w, VariableArray<bool> y)
    { // Create x vector, augmented by 1 
        Range j = y.Range; 
        Vector[] xdata = new Vector[incomes.Length];
        double noise = 0.1;
        for (int i = 0; i < xdata.Length; i++)
            xdata[i] = Vector.FromArray(incomes[i], ages[i], 1);
        VariableArray<Vector> x = Variable.Observed(xdata, j); // Bayes Point Machine double noise = 0.1;  
        y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]), noise) > 0;
    }

    public static void Main(string[] args)
    {
        //FairTwoCoin(0.5);
        //Gaussian(0, 1);
        //LargeSampling();
    }
}