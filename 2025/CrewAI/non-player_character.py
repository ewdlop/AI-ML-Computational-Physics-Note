import asyncio
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Define AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Define an NPC AI Agent
npc_ai = Agent(
    role="NPC AI",
    goal="Generate dynamic dialogue for players",
    backstory="An intelligent game character capable of responding based on player actions.",
    llm=llm,
    verbose=True
)

# Asynchronous function to process AI task
async def generate_npc_dialogue(query):
    task = Task(query, agent=npc_ai)
    crew = Crew(agents=[npc_ai], tasks=[task])
    response = await asyncio.to_thread(crew.kickoff)
    return response

# Example Usage
async def main():
    query = "Tell the player a legend about this town."
    npc_response = await generate_npc_dialogue(query)
    print("NPC:", npc_response)

asyncio.run(main())
