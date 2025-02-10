import time
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Define AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Create AI Agent
test_ai = Agent(
    role="Test Agent",
    goal="Perform AI-based tasks",
    llm=llm,
    verbose=True
)

# Measure execution time
def benchmark_ai(query):
    start_time = time.time()
    task = Task(query, agent=test_ai)
    crew = Crew(agents=[test_ai], tasks=[task])
    response = crew.kickoff()
    end_time = time.time()
    
    return response, end_time - start_time

# Example Usage
query = "Generate a short story about a knight."
response, exec_time = benchmark_ai(query)
print(f"AI Response: {response}\nExecution Time: {exec_time:.2f} seconds")
