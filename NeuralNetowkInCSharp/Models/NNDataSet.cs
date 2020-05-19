using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetowkInCSharp.Models
{
    public class NNDataSet
    {
        public double[] Values { get; set; }
        public double[] Targets { get; set; }
        public NNDataSet(double[] values, double[] targets)
        {
            Values = values;
            Targets = targets;
        }
    }
}
