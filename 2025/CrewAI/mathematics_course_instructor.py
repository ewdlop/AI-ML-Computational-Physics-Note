from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# 1️⃣ Math Instructor AI
math_instructor = Agent(
    role="Mathematics Instructor",
    goal="Teach college mathematics using Coq, ensuring rigorous formal proofs.",
    backstory="A professor specializing in theorem proving, using Coq to verify mathematical statements.",
    llm=llm,
    verbose=True
)

# 2️⃣ Proof Validator AI
proof_validator = Agent(
    role="Coq Proof Validator",
    goal="Verify student-submitted Coq proofs and provide feedback.",
    backstory="A Coq expert that checks if mathematical proofs are formally correct.",
    llm=llm,
    verbose=True
)

# 3️⃣ Problem Generator AI
problem_generator = Agent(
    role="Math Problem Generator",
    goal="Generate new Coq-based proof exercises for students.",
    backstory="A problem-creating AI that generates formal logic and calculus exercises in Coq.",
    llm=llm,
    verbose=True
)

# 4️⃣ Course Assistant AI
course_assistant = Agent(
    role="Course Assistant",
    goal="Convert Coq proofs into LaTeX for easy reading.",
    backstory="An AI assistant that formats Coq proofs into well-structured LaTeX documents.",
    llm=llm,
    verbose=True
)

# Define tasks
tasks = [
    Task("Explain the proof of the fundamental theorem of calculus using Coq.", agent=math_instructor),
    Task("Verify a student-submitted proof of a real analysis theorem in Coq.", agent=proof_validator),
    Task("Generate a new Coq-based exercise on group theory.", agent=problem_generator),
    Task("Convert a Coq proof of a number theory theorem into LaTeX format.", agent=course_assistant),
]

# Create the Crew
crew = Crew(agents=[math_instructor, proof_validator, problem_generator, course_assistant], tasks=tasks)

# Run the Crew
crew.kickoff()
