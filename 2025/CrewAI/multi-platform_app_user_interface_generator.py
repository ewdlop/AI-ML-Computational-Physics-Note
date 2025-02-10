import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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

# Function to generate the component
def generate_component():
    component = component_var.get()
    animation = animation_var.get()
    color = color_var.get()
    output_file = filedialog.asksaveasfilename(defaultextension=".cs", filetypes=[("C# Files", "*.cs")])

    if not output_file:
        return

    task_description = f"Generate a .NET MAUI {component} with background color {color}."
    if animation == "Yes":
        task_description += " Include an interactive animation effect."

    # Define AI Task
    task = Task(task_description, agent=maui_generator)

    # Create the Crew
    crew = Crew(agents=[maui_generator], tasks=[task])

    # Run AI Task
    generated_code = crew.kickoff()

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_code)

    messagebox.showinfo("Success", f"Component generated successfully: {output_file}")

# GUI Setup
root = tk.Tk()
root.title("MAUI Component Generator")
root.geometry("400x300")

# Component Selection
ttk.Label(root, text="Choose a .NET MAUI Component:").pack(pady=5)
component_var = tk.StringVar(value="Button")
component_dropdown = ttk.Combobox(root, textvariable=component_var, values=["Button", "Label", "Entry", "CustomView"])
component_dropdown.pack(pady=5)

# Animation Selection
ttk.Label(root, text="Include Animation?").pack(pady=5)
animation_var = tk.StringVar(value="No")
animation_dropdown = ttk.Combobox(root, textvariable=animation_var, values=["Yes", "No"])
animation_dropdown.pack(pady=5)

# Color Selection
ttk.Label(root, text="Set Background Color:").pack(pady=5)
color_var = tk.StringVar(value="Blue")
color_entry = ttk.Entry(root, textvariable=color_var)
color_entry.pack(pady=5)

# Generate Button
generate_button = ttk.Button(root, text="Generate Component", command=generate_component)
generate_button.pack(pady=10)

# Run GUI
root.mainloop()
