using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Orchestration;
using Microsoft.SemanticKernel.SkillDefinition;

KernelBuilder builder = new KernelBuilder();
builder.WithOpenAIChatCompletionService(
         "gpt-3.5-turbo",// OpenAI Model name
         "...your OpenAI API Key...");// OpenAI API Key

IKernel kernel = builder.Build();C

{

string prompt = @"{{$input}}

One line TLDR with the fewest words.";

ISKFunction summarize = kernel.CreateSemanticFunction(prompt, maxTokens: 100);

string text1 = @"
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.";

string text2 = @"
1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
2. The acceleration of an object depends on the mass of the object and the amount of force applied.
3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first.";

Console.WriteLine(await summarize.InvokeAsync(text1));
Console.WriteLine(await summarize.InvokeAsync(text2));
}

{
    string translationPrompt = @"{{$input}}

Translate the text to math.";

    string summarizePrompt = @"{{$input}}

Give me a TLDR with the fewest words.";

    ISKFunction translator = kernel.CreateSemanticFunction(translationPrompt, maxTokens: 200);
    ISKFunction summarize = kernel.CreateSemanticFunction(summarizePrompt, maxTokens: 100);

    string inputText = @"
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.";

    // Run two prompts in sequence (prompt chaining)
    SKContext output = await kernel.RunAsync(inputText, translator, summarize);

    Console.WriteLine(output);

    // Output: ΔE = 0, ΔSuniv > 0, S = 0 at 0K.
}
