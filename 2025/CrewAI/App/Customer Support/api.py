from flask import Flask, request, jsonify
from flask_limiter import Limiter
import redis
from celery import Celery
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)
limiter = Limiter(app, key_func=lambda: request.remote_addr, default_limits=["5 per second"])
cache = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

# Define AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Create AI Agent
api_ai = Agent(
    role="API AI",
    goal="Answer user queries instantly",
    backstory="A customer support AI for real-time responses.",
    llm=llm,
    verbose=True
)

@celery_app.task
def process_ai_request(query):
    if cache.exists(query):
        return cache.get(query).decode('utf-8')

    task = Task(query, agent=api_ai)
    crew = Crew(agents=[api_ai], tasks=[task])
    response = crew.kickoff()

    cache.set(query, response, ex=3600)  # Cache for 1 hour
    return response

@app.route("/ask", methods=["POST"])
@limiter.limit("10 per minute")  # Rate limit AI requests
def ask_ai():
    user_query = request.json["query"]
    task = process_ai_request.delay(user_query)
    return jsonify({"task_id": task.id})

if __name__ == "__main__":
    app.run(debug=True)
