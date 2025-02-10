from flask import Flask, request, jsonify
from celery import Celery
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import asyncio

# Initialize Flask app
app = Flask(__name__)

# Configure Celery
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

# Define AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Create AI Agent
support_ai = Agent(
    role="Customer Support AI",
    goal="Assist customers with real-time support",
    backstory="A virtual assistant trained to answer customer queries.",
    llm=llm,
    verbose=True
)

@celery.task
def process_customer_request(query):
    task = Task(query, agent=support_ai)
    crew = Crew(agents=[support_ai], tasks=[task])
    return crew.kickoff()

@app.route("/ask", methods=["POST"])
async def ask_ai():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # For I/O-bound tasks, use asyncio
    result = await asyncio.to_thread(process_customer_request.apply_async, args=[user_query])
    return jsonify({"task_id": result.id}), 202

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = process_customer_request.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)  # Exception info
        }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
