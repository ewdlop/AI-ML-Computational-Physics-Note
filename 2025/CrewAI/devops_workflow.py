from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# 1️⃣ Code Generator AI
code_generator = Agent(
    role="AI Developer",
    goal="Write high-quality Python code based on requirements.",
    backstory="A highly skilled AI software engineer with expertise in multiple programming languages.",
    llm=llm,
    verbose=True
)

# 2️⃣ Code Reviewer AI
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Review generated code for correctness, efficiency, and best practices.",
    backstory="An experienced software architect with a deep understanding of clean coding principles.",
    llm=llm,
    verbose=True
)

# 3️⃣ Code Tester AI
code_tester = Agent(
    role="Testing Engineer",
    goal="Generate and run unit tests to verify the correctness of the code.",
    backstory="An expert in software testing, ensuring the reliability of code using unit tests and debugging.",
    llm=llm,
    verbose=True
)

# 4️⃣ CI/CD Agent
cicd_agent = Agent(
    role="Deployment Engineer",
    goal="Ensure the code is deployed correctly using CI/CD pipelines.",
    backstory="A DevOps engineer skilled in automating deployments and ensuring reliability in production.",
    llm=llm,
    verbose=True
)

# Define tasks
tasks = [
    Task("Generate Python code for a REST API endpoint.", agent=code_generator),
    Task("Review the generated code and suggest improvements.", agent=code_reviewer),
    Task("Write unit tests for the API endpoint.", agent=code_tester),
    Task("Deploy the API endpoint using Docker and CI/CD pipeline.", agent=cicd_agent),
]

# Create the Crew
crew = Crew(agents=[code_generator, code_reviewer, code_tester, cicd_agent], tasks=tasks)

# Run the Crew
crew.kickoff()
