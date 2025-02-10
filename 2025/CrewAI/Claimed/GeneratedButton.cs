using Microsoft.Maui.Controls;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using System.Text;

[Generator]
public class GeneratedButton : ISourceGenerator
{
    public void Initialize(GeneratorInitializationContext context)
    {
        context.RegisterForSyntaxNotifications(() => new ClassSyntaxReceiver());
    }

    public void Execute(GeneratorExecutionContext context)
    {
        var generatedCode = GenerateButtonCode();
        context.AddSource("GeneratedButton.g.cs", SourceText.From(generatedCode, Encoding.UTF8));
    }

    private string GenerateButtonCode()
    {
        return @"
using Microsoft.Maui.Controls;

public partial class AnimatedButton : Button
{
    public AnimatedButton()
    {
        Text = ""Click Me"";
        BackgroundColor = Colors.Red;
        Clicked += (sender, args) =>
        {
            this.ScaleTo(1.2, 250);
            this.FadeTo(0.5, 250);
            this.ScaleTo(1.0, 250);
            this.FadeTo(1.0, 250);
        };
    }
}";
    }
}

class ClassSyntaxReceiver : ISyntaxReceiver
{
    public void OnVisitSyntaxNode(SyntaxNode syntaxNode) { }
}
