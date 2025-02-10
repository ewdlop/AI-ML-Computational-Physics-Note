from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# 1️⃣ C# Code Generator AI
csharp_generator = Agent(
    role="C# Source Generator",
    goal="Generate C# Source Generators using Roslyn for compile-time code automation.",
    backstory="An expert in C# Source Generators, leveraging Roslyn to generate high-performance C# code at compile time.",
    llm=llm,
    verbose=True
)

# 2️⃣ Code Reviewer AI
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Analyze the generated C# source generator code for performance, correctness, and adherence to best practices.",
    backstory="A senior C# developer and architect who reviews code for efficiency, maintainability, and security.",
    llm=llm,
    verbose=True
)

# 3️⃣ Testing AI
test_agent = Agent(
    role="Unit Test Generator",
    goal="Write unit tests to validate the generated C# code using xUnit or NUnit.",
    backstory="A software testing specialist ensuring reliability and correctness using automated unit testing frameworks.",
    llm=llm,
    verbose=True
)

# 4️⃣ CI/CD Deployment AI
deployment_agent = Agent(
    role="CI/CD Engineer",
    goal="Prepare and deploy the C# Source Generator to NuGet and integrate it into a CI/CD pipeline.",
    backstory="A DevOps expert specializing in deploying .NET projects using GitHub Actions and Azure Pipelines.",
    llm=llm,
    verbose=True
)

# Define tasks
tasks = [
    Task("Generate a C# Source Generator that automatically implements INotifyPropertyChanged.", agent=csharp_generator),
    Task("Review the generated source generator for performance and maintainability.", agent=code_reviewer),
    Task("Generate unit tests to verify the functionality of the generated code.", agent=test_agent),
    Task("Prepare the source generator for deployment using GitHub Actions and publish it to NuGet.", agent=deployment_agent),
]

# Create the Crew
crew = Crew(agents=[csharp_generator, code_reviewer, test_agent, deployment_agent], tasks=tasks)

# Run the Crew
crew.kickoff()
