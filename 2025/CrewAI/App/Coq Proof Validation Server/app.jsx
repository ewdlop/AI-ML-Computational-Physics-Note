import React, { useState, useEffect, useRef } from "react";
import io from "socket.io-client";

const socket = io("http://localhost:5000");

function App() {
  const [proof, setProof] = useState("");
  const [studentName, setStudentName] = useState("");
  const [feedback, setFeedback] = useState("");
  const [grade, setGrade] = useState(null);
  const [videoChatActive, setVideoChatActive] = useState(false);
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  let peerConnection = null;

  useEffect(() => {
    socket.on("webrtc_offer", async (data) => {
      peerConnection = new RTCPeerConnection();
      peerConnection.setRemoteDescription(new RTCSessionDescription(data));
      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);
      socket.emit("webrtc_answer", answer);
    });

    socket.on("webrtc_answer", (data) => {
      peerConnection.setRemoteDescription(new RTCSessionDescription(data));
    });

    socket.on("webrtc_ice_candidate", (data) => {
      peerConnection.addIceCandidate(new RTCIceCandidate(data));
    });
  }, []);

  const submitProof = async () => {
    const res = await fetch("http://localhost:5000/submit_proof", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ student_name: studentName, proof }),
    });
    const data = await res.json();
    setFeedback(data.feedback);
    setGrade(data.grade);
  };

  const startVideoChat = async () => {
    setVideoChatActive(true);
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    localVideoRef.current.srcObject = stream;

    peerConnection = new RTCPeerConnection();
    stream.getTracks().forEach((track) => peerConnection.addTrack(track, stream));

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit("webrtc_offer", offer);
  };

  return (
    <div>
      <h1>AI-Powered Mathematics Course</h1>
      
      <div>
        <h2>Submit a Coq Proof</h2>
        <input type="text" placeholder="Your Name" value={studentName} onChange={(e) => setStudentName(e.target.value)} />
        <textarea value={proof} onChange={(e) => setProof(e.target.value)} rows="4" cols="50" />
        <button onClick={submitProof}>Submit</button>
        <p><strong>AI Feedback:</strong> {feedback}</p>
        <p><strong>Grade:</strong> {grade ? `${grade}/10` : "Pending"}</p>
      </div>

      <div>
        <h2>Live Video Chat</h2>
        <button onClick={startVideoChat}>Start Video Chat</button>
        {videoChatActive && (
          <div>
            <video ref={localVideoRef} autoPlay playsInline muted style={{ width: "300px" }} />
            <video ref={remoteVideoRef} autoPlay playsInline style={{ width: "300px" }} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
