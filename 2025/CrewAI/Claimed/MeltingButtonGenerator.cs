using Microsoft.Maui.Controls;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;
using System.Text;

[Generator]
public class MeltingButtonGenerator : ISourceGenerator
{
    public void Initialize(GeneratorInitializationContext context)
    {
        context.RegisterForSyntaxNotifications(() => new ClassSyntaxReceiver());
    }

    public void Execute(GeneratorExecutionContext context)
    {
        if (context.SyntaxReceiver is not ClassSyntaxReceiver receiver)
            return;

        foreach (var classDeclaration in receiver.Classes)
        {
            var className = classDeclaration.Identifier.Text;
            var generatedCode = GenerateMeltingButtonComponent(className);
            context.AddSource($"{className}_MeltingButton.g.cs", SourceText.From(generatedCode, Encoding.UTF8));
        }
    }

    private string GenerateMeltingButtonComponent(string className)
    {
        return $@"
using Microsoft.Maui.Controls;

public partial class {className} : ContentView
{{
    public {className}()
    {{
        Button meltingButton = new Button
        {{
            Text = ""Melting Keys"",
            BackgroundColor = Colors.Blue,
            TextColor = Colors.White
        }};
        
        meltingButton.Clicked += (sender, args) =>
        {{
            meltingButton.ScaleTo(1.2, 500);
            meltingButton.FadeTo(0.5, 500);
            meltingButton.ScaleTo(1.0, 500);
            meltingButton.FadeTo(1.0, 500);
        }};

        Content = new StackLayout
        {{
            Children = {{ meltingButton }}
        }};
    }}
}}";
    }
}

class ClassSyntaxReceiver : ISyntaxReceiver
{
    public List<ClassDeclarationSyntax> Classes { get; } = new List<ClassDeclarationSyntax>();

    public void OnVisitSyntaxNode(SyntaxNode syntaxNode)
    {
        if (syntaxNode is ClassDeclarationSyntax classDeclaration)
        {
            Classes.Add(classDeclaration);
        }
    }
}
