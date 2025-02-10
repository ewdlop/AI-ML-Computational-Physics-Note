from flask import Flask, request, jsonify
from crewai import Agent, Crew, Task
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)

# AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# User Support Agent
support_ai = Agent(
    role="User Support AI",
    goal="Help users with product issues",
    llm=llm,
    verbose=True
)

# API Route
@app.route("/ask", methods=["POST"])
def ask_ai():
    user_query = request.json["query"]
    task = Task(user_query, agent=support_ai)
    crew = Crew(agents=[support_ai], tasks=[task])
    response = crew.kickoff()
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
