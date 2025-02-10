from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# 1️⃣ .NET MAUI UI Component Generator AI
maui_generator = Agent(
    role="MAUI Component Generator",
    goal="Generate .NET MAUI UI components with XAML and C# using Roslyn Source Generators.",
    backstory="An expert in .NET MAUI UI development and Roslyn Source Generators.",
    llm=llm,
    verbose=True
)

# 2️⃣ Code Reviewer AI
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Analyze the generated MAUI component for performance, usability, and best practices.",
    backstory="A senior C# developer specializing in .NET UI frameworks.",
    llm=llm,
    verbose=True
)

# 3️⃣ UI Testing AI
ui_test_agent = Agent(
    role="UI Testing Engineer",
    goal="Write unit tests for the .NET MAUI UI component using MSTest.",
    backstory="An expert in UI testing and automation for .NET applications.",
    llm=llm,
    verbose=True
)

# 4️⃣ CI/CD Deployment AI
deployment_agent = Agent(
    role="CI/CD Engineer",
    goal="Deploy the MAUI UI component as a NuGet package and integrate CI/CD.",
    backstory="A DevOps engineer specializing in .NET deployments using GitHub Actions and NuGet.",
    llm=llm,
    verbose=True
)

# Define tasks
tasks = [
    Task("Generate a .NET MAUI Button with a 'melting keys' animation effect.", agent=maui_generator),
    Task("Review the generated component for performance and best practices.", agent=code_reviewer),
    Task("Write UI tests for the MAUI component using MSTest.", agent=ui_test_agent),
    Task("Prepare the MAUI component for NuGet deployment using GitHub Actions.", agent=deployment_agent),
]

# Create the Crew
crew = Crew(agents=[maui_generator, code_reviewer, ui_test_agent, deployment_agent], tasks=tasks)

# Run the Crew
crew.kickoff()
