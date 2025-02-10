from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI

# Define LLM model
llm = ChatOpenAI(model="gpt-4-turbo")

# 1️⃣ Game Master AI (Narrator)
game_master = Agent(
    role="Game Master",
    goal="Create dynamic quests and lore based on player actions",
    backstory="A master storyteller who adjusts the world based on real-time gameplay.",
    llm=llm,
    verbose=True
)

# 2️⃣ Enemy AI
enemy_ai = Agent(
    role="Enemy AI",
    goal="Control enemy difficulty based on player progress",
    backstory="An adaptive combat strategist controlling enemy behavior.",
    llm=llm,
    verbose=True
)

# 3️⃣ NPC AI (Companion or Quest Giver)
npc_ai = Agent(
    role="NPC AI",
    goal="Generate realistic conversations and interact with the player",
    backstory="A helpful NPC who provides quests and guidance.",
    llm=llm,
    verbose=True
)

# Define tasks
tasks = [
    Task("Generate a new quest for the player", agent=game_master),
    Task("Adjust enemy tactics based on player skill", agent=enemy_ai),
    Task("Create interactive NPC dialogue", agent=npc_ai),
]

# Create CrewAI team
crew = Crew(agents=[game_master, enemy_ai, npc_ai], tasks=tasks)

# Start game AI
crew.kickoff()
