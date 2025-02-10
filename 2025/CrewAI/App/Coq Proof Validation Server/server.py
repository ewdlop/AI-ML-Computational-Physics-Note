from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://user:password@localhost/mathcourse"
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# AI Model
llm = ChatOpenAI(model="gpt-4-turbo")

# Database Model for Proof Submissions
class ProofSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_name = db.Column(db.String(100), nullable=False)
    proof_code = db.Column(db.Text, nullable=False)
    ai_feedback = db.Column(db.Text, nullable=True)
    grade = db.Column(db.Float, nullable=True)

db.create_all()

# AI Agents
proof_validator = Agent(
    role="Coq Proof Validator",
    goal="Check student-submitted Coq proofs and assign a grade.",
    llm=llm,
    verbose=True
)

# WebRTC Signaling (for Live Video Chat)
@socketio.on("webrtc_offer")
def handle_webrtc_offer(data):
    emit("webrtc_offer", data, broadcast=True)

@socketio.on("webrtc_answer")
def handle_webrtc_answer(data):
    emit("webrtc_answer", data, broadcast=True)

@socketio.on("webrtc_ice_candidate")
def handle_ice_candidate(data):
    emit("webrtc_ice_candidate", data, broadcast=True)

# AI-Graded Proof Submission
@app.route("/submit_proof", methods=["POST"])
def submit_proof():
    student_name = request.json["student_name"]
    proof_code = request.json["proof"]

    # AI task for grading
    task = Task(f"Evaluate this Coq proof, provide feedback, and assign a grade: {proof_code}", agent=proof_validator)
    crew = Crew(agents=[proof_validator], tasks=[task])
    ai_response = crew.kickoff()

    # Parse AI response (example format: "Grade: 8.5/10 - Feedback: ...")
    grade = float(ai_response.split("Grade:")[1].split("/")[0].strip()) if "Grade:" in ai_response else None
    feedback = ai_response.split("Feedback:")[1].strip() if "Feedback:" in ai_response else ai_response

    # Store in database
    submission = ProofSubmission(student_name=student_name, proof_code=proof_code, ai_feedback=feedback, grade=grade)
    db.session.add(submission)
    db.session.commit()

    return jsonify({"feedback": feedback, "grade": grade})

if __name__ == "__main__":
    socketio.run(app, debug=True)
