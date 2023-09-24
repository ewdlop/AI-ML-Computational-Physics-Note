using Azure.AI.OpenAI;
using Azure;
using System.Text.Json;
{
    bool useAzureOpenAI = false;

    OpenAIClient client = useAzureOpenAI
        ? new OpenAIClient(
            new Uri("https://your-azure-openai-resource.com/"),
            new AzureKeyCredential("your-azure-openai-resource-api-key"))
        : new OpenAIClient("your-api-key-from-platform.openai.com");

    List<string> examplePrompts = new(){
        "How are you today?",
        "What is Azure OpenAI?",
        "Why do children love dinosaurs?",
        "Generate a proof of Euler's identity",
        "Describe in single words only the good things that come into your mind about your mother.",
    };
    {
        string deploymentName = "text-davinci-003";

        foreach (string prompt in examplePrompts)
        {
            Console.Write($"Input: {prompt}");
            CompletionsOptions completionsOptions = new CompletionsOptions();
            completionsOptions.Prompts.Add(prompt);

            Response<Completions> completionsResponse = client.GetCompletions(deploymentName, completionsOptions);
            string completion = completionsResponse.Value.Choices[0].Text;
            Console.WriteLine($"Chatbot: {completion}");
        }
    }
    {
        string textToSummarize = 
        """
            Two independent experiments reported their results this morning at CERN, Europe's high-energy physics laboratory near Geneva in Switzerland. Both show convincing evidence of a new boson particle weighing around 125 gigaelectronvolts, which so far fits predictions of the Higgs previously made by theoretical physicists.

            ""As a layman I would say: 'I think we have it'. Would you agree?"" Rolf-Dieter Heuer, CERN's director-general, asked the packed auditorium. The physicists assembled there burst into applause.
            :"
        """;

        string summarizationPrompt = $"""""""
            Summarize the following text.

            Text:
            """"""
            {textToSummarize}
            """"""

            Summary:
        """"""";

        Console.Write($"Input: {summarizationPrompt}");
        var completionsOptions = new CompletionsOptions()
        {
            Prompts = { summarizationPrompt },
        };

        string deploymentName = "text-davinci-003";

        Response<Completions> completionsResponse = client.GetCompletions(deploymentName, completionsOptions);
        string completion = completionsResponse.Value.Choices[0].Text;
        Console.WriteLine($"Summarization: {completion}");
    }
    {
        ChatCompletionsOptions chatCompletionsOptions = new ChatCompletionsOptions()
        {
            Messages =
            {
                new ChatMessage(ChatRole.System, "You are a helpful assistant. You will talk like a pirate."),
                new ChatMessage(ChatRole.User, "Can you help me?"),
                new ChatMessage(ChatRole.Assistant, "Arrrr! Of course, me hearty! What can I do for ye?"),
                new ChatMessage(ChatRole.User, "What's the best way to train a parrot?"),
            }
        };

        Response<StreamingChatCompletions> response = await client.GetChatCompletionsStreamingAsync(
            deploymentOrModelName: "gpt-3.5-turbo",
            chatCompletionsOptions);
        using StreamingChatCompletions streamingChatCompletions = response.Value;

        await foreach (StreamingChatChoice choice in streamingChatCompletions.GetChoicesStreaming())
        {
            await foreach (ChatMessage message in choice.GetMessageStreaming())
            {
                Console.Write(message.Content);
            }
            Console.WriteLine();
        }
    }
    {
        FunctionDefinition getWeatherFuntionDefinition = new FunctionDefinition()
        {
            Name = "get_current_weather",
            Description = "Get the current weather in a given location",
            Parameters = BinaryData.FromObjectAsJson(
    new
                {
                    Type = "object",
                    Properties = new
                    {
                        Location = new
                        {
                            Type = "string",
                            Description = "The city and state, e.g. San Francisco, CA",
                        },
                        Unit = new
                        {
                            Type = "string",
                            Enum = new[] { "celsius", "fahrenheit" },
                        }
                    },
                    Required = new[] { "location" },
                },
            new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }),
        };
        List<ChatMessage> conversationMessages = new List<ChatMessage>()
        {
            new(ChatRole.User, "What is the weather like in Boston?"),
        };

        ChatCompletionsOptions chatCompletionsOptions = new ChatCompletionsOptions();
        foreach (ChatMessage chatMessage in conversationMessages)
        {
            chatCompletionsOptions.Messages.Add(chatMessage);
        }
        chatCompletionsOptions.Functions.Add(getWeatherFuntionDefinition);

        Response<ChatCompletions> response = await client.GetChatCompletionsAsync(
            "gpt-35-turbo-0613",
            chatCompletionsOptions);

        ChatChoice responseChoice = response.Value.Choices[0];
        if (responseChoice.FinishReason == CompletionsFinishReason.FunctionCall)
        {
            // Include the FunctionCall message in the conversation history
            conversationMessages.Add(responseChoice.Message);

            if (responseChoice.Message.FunctionCall.Name == "get_current_weather")
            {
                // Validate and process the JSON arguments for the function call
                string unvalidatedArguments = responseChoice.Message.FunctionCall.Arguments;
                object? functionResultData = null; // GetYourFunctionResultData(unvalidatedArguments);
                                                       // Here, replacing with an example as if returned from GetYourFunctionResultData
                functionResultData = new
                {
                    Temperature = 31,
                    Unit = "celsius",
                };
                // Serialize the result data from the function into a new chat message with the 'Function' role,
                // then add it to the messages after the first User message and initial response FunctionCall
                ChatMessage? functionResponseMessage = new ChatMessage(
                    ChatRole.Function,
                    JsonSerializer.Serialize(
                        functionResultData,
                        new JsonSerializerOptions() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
                conversationMessages.Add(functionResponseMessage);
                // Now make a new request using all three messages in conversationMessages
            }
        }
    }
    {
        ChatCompletionsOptions chatCompletionsOptions = new ChatCompletionsOptions()
        {
            Messages =
            {
                new ChatMessage(
                    ChatRole.System,
                    "You are a helpful assistant that answers questions about the Contoso product database."),
                new ChatMessage(ChatRole.User, "What are the best-selling Contoso products this month?")
            },
            // The addition of AzureChatExtensionsOptions enables the use of Azure OpenAI capabilities that add to
            // the behavior of Chat Completions, here the "using your own data" feature to supplement the context
            // with information from an Azure Cognitive Search resource with documents that have been indexed.
            AzureExtensionsOptions = new AzureChatExtensionsOptions()
            {
                Extensions =
                {
                    new AzureCognitiveSearchChatExtensionConfiguration()
                    {
                        SearchEndpoint = new Uri("https://your-contoso-search-resource.search.windows.net"),
                        IndexName = "contoso-products-index",
                        SearchKey = new AzureKeyCredential("<your Cognitive Search resource API key>"),
                    }
                }
            }
        };
        Response<ChatCompletions> response = await client.GetChatCompletionsAsync(
            "gpt-35-turbo-0613",
            chatCompletionsOptions);
        ChatMessage message = response.Value.Choices[0].Message;
        // The final, data-informed response still appears in the ChatMessages as usual
        Console.WriteLine($"{message.Role}: {message.Content}");
        // Responses that used extensions will also have Context information that includes special Tool messages
        // to explain extension activity and provide supplemental information like citations.
        Console.WriteLine($"Citations and other information:");
        foreach (ChatMessage contextMessage in message.AzureExtensionsContext.Messages)
        {
            // Note: citations and other extension payloads from the "tool" role are often encoded JSON documents
            // and need to be parsed as such; that step is omitted here for brevity.
            Console.WriteLine($"{contextMessage.Role}: {contextMessage.Content}");
        }
    }
    {
        Response<ImageGenerations> imageGenerations = await client.GetImageGenerationsAsync(
        new ImageGenerationOptions()
        {
            Prompt = "a happy monkey eating a banana, in watercolor",
            Size = ImageSize.Size256x256,
        });

        // Image Generations responses provide URLs you can use to retrieve requested images
        Uri imageUri = imageGenerations.Value.Data[0].Url;
    }
}
object? GetYourFunctionResultData(string unvalidatedArguments) => null;