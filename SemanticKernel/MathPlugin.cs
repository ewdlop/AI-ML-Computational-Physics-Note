using Microsoft.SemanticKernel.Orchestration;
using Microsoft.SemanticKernel.SkillDefinition;
using System.ComponentModel;
using System.Globalization;

namespace Plugins;

public static class MathPlugin
{
    [SKFunction, Description("Take the square root of a number")]
    public static string Sqrt(string input)
    {
        return Math.Sqrt(Convert.ToDouble(input, CultureInfo.InvariantCulture)).ToString(CultureInfo.InvariantCulture);
    }

    [SKFunction, Description("Add two numbers")]
    [SKParameter("input", "The first number to add")]
    [SKParameter("number2", "The second number to add")]
    public static string Add(this SKContext context)
    {
        return (
            Convert.ToDouble(context.Variables["input"], CultureInfo.InvariantCulture) +
            Convert.ToDouble(context.Variables["number2"], CultureInfo.InvariantCulture)
        ).ToString(CultureInfo.InvariantCulture);
    }

    [SKFunction, Description("Subtract two numbers")]
    [SKParameter("input", "The first number to subtract from")]
    [SKParameter("number2", "The second number to subtract away")]
    public static string Subtract(this SKContext context)
    {
        return (
            Convert.ToDouble(context.Variables["input"], CultureInfo.InvariantCulture) -
            Convert.ToDouble(context.Variables["number2"], CultureInfo.InvariantCulture)
        ).ToString(CultureInfo.InvariantCulture);
    }

    [SKFunction, Description("Multiply two numbers. When increasing by a percentage, don't forget to add 1 to the percentage.")]
    [SKParameter("input", "The first number to multiply")]
    [SKParameter("number2", "The second number to multiply")]
    public static string Multiply(this SKContext context)
    {
        return (
            Convert.ToDouble(context.Variables["input"], CultureInfo.InvariantCulture) *
            Convert.ToDouble(context.Variables["number2"], CultureInfo.InvariantCulture)
        ).ToString(CultureInfo.InvariantCulture);
    }

    [SKFunction, Description("Divide two numbers")]
    [SKParameter("input", "The first number to divide from")]
    [SKParameter("number2", "The second number to divide by")]
    public static string Divide(this SKContext context)
    {
        return (
            Convert.ToDouble(context.Variables["input"], CultureInfo.InvariantCulture) /
            Convert.ToDouble(context.Variables["number2"], CultureInfo.InvariantCulture)
        ).ToString(CultureInfo.InvariantCulture);
    }
}