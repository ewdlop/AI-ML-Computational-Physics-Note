from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI

# Define LLM model
llm = ChatOpenAI(model="gpt-4-turbo")

# 1️⃣ API Manager
api_manager = Agent(
    role="API Manager",
    goal="Handle API requests and ensure smooth communication between frontend and backend",
    backstory="A backend engineer responsible for managing API endpoints.",
    llm=llm,
    verbose=True
)

# 2️⃣ Database Agent
database_agent = Agent(
    role="Database Query Handler",
    goal="Optimize and execute SQL queries efficiently",
    backstory="An expert in relational databases, ensuring fast query execution.",
    llm=llm,
    verbose=True
)

# 3️⃣ User Support AI
support_ai = Agent(
    role="User Support Assistant",
    goal="Answer customer questions and assist with common troubleshooting issues",
    backstory="A customer service chatbot that provides helpful responses in real time.",
    llm=llm,
    verbose=True
)

# Define tasks
tasks = [
    Task("Process API requests and handle data transmission", agent=api_manager),
    Task("Optimize and fetch data from the database", agent=database_agent),
    Task("Assist users with real-time queries", agent=support_ai),
]

# Create CrewAI full-stack team
crew = Crew(agents=[api_manager, database_agent, support_ai], tasks=tasks)

# Start the full-stack AI
crew.kickoff()
