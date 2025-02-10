import click
import os
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Define .NET MAUI Generator AI Agent
maui_generator = Agent(
    role="MAUI Component Generator",
    goal="Generate .NET MAUI UI components dynamically using C# Source Generators.",
    backstory="A senior .NET MAUI developer specializing in UI automation and Roslyn.",
    llm=llm,
    verbose=True
)

@click.command()
@click.option("--component", type=click.Choice(["Button", "Label", "Entry", "CustomView"]), prompt="Choose a .NET MAUI Component")
@click.option("--animation", is_flag=True, help="Add animation effects?")
@click.option("--color", default="Blue", help="Set component background color")
@click.option("--output", default="GeneratedMauiComponent.cs", help="Output C# file")
def generate_maui_component(component, animation, color, output):
    """CLI tool to generate .NET MAUI components with Source Generators"""
    
    # Task description
    task_description = f"Generate a .NET MAUI {component} with background color {color}."
    if animation:
        task_description += " Include an interactive animation effect."
    
    # Define the AI Task
    task = Task(task_description, agent=maui_generator)

    # Create the Crew
    crew = Crew(agents=[maui_generator], tasks=[task])

    # Run AI Task
    generated_code = crew.kickoff()

    # Save to file
    with open(output, "w", encoding="utf-8") as f:
        f.write(generated_code)

    click.echo(f"✅ Component generated: {output}")

if __name__ == "__main__":
    generate_maui_component()
