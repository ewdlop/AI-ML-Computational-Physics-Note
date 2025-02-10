import asyncio
import websockets
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

# Define AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Create NPC AI Agent
npc_ai = Agent(
    role="NPC AI",
    goal="Provide dynamic conversations for players",
    backstory="An intelligent game character who adapts to the player's dialogue.",
    llm=llm,
    verbose=True
)

# WebSocket server for real-time AI interaction
async def npc_server(websocket, path):
    async for message in websocket:
        print(f"Player: {message}")

        # Asynchronously generate AI response
        task = Task(message, agent=npc_ai)
        crew = Crew(agents=[npc_ai], tasks=[task])
        response = await asyncio.to_thread(crew.kickoff)

        await websocket.send(response)
        print(f"NPC: {response}")

# Start WebSocket Server (Runs on ws://localhost:8765)
start_server = websockets.serve(npc_server, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
