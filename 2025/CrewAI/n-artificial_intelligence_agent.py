import concurrent.futures
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Define AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Create AI Agents
agents = [
    Agent(role=f"AI-{i}", goal="Process user queries", llm=llm, verbose=True)
    for i in range(5)  # Create 5 AI agents
]

def process_request(query, agent):
    task = Task(query, agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    return crew.kickoff()

queries = ["Tell me a story.", "Describe a castle.", "What is AI?", "Explain gravity.", "Write a poem."]

# Run AI tasks in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_request, queries, agents))

# Display AI responses
for i, res in enumerate(results):
    print(f"Query {i+1}: {queries[i]}\nResponse: {res}\n")
