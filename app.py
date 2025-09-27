from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import webbrowser

app = FastAPI()

# CORS middleware for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store latest pose data
latest = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Camera Smoke Test</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.0/dist/pose-detection.min.js"></script>
</head>
<body>
    <h1>Camera Smoke Test</h1>
    <div id="status" style="color: green;">Ready</div>
    <button id="startBtn" onclick="window.__start()">Start Camera</button>
    <div id="hud" style="margin-top:8px;">L: <span id="lAng">--</span>° | R: <span id="rAng">--</span>°</div>
    <br><br>
    <div id="stack" style="position:relative;width:640px;height:480px;">
      <video id="video" width="640" height="480" autoplay muted playsinline style="display:block;position:absolute;left:0;top:0;z-index:1;"></video>
      <canvas id="overlay" width="640" height="480" style="position:absolute;left:0;top:0;pointer-events:none;z-index:2;"></canvas>
    </div>
    
    <script>
    window.addEventListener('error', e => { const s=document.getElementById('status'); if(s) s.textContent='JS error: '+e.message; });
    
    let sendFrame = 0, lastPost = 0;
    const SEND_MS = 50; // ~20 Hz
    
    async function sendAngles(leftAngle, rightAngle, leftScore, rightScore){
      const now = Date.now();
      if (now - lastPost < SEND_MS) return;
      lastPost = now;
      const payload = {
        frameId: sendFrame++,
        ts: now/1000,
        leftKneeAngle: leftAngle ?? null,
        rightKneeAngle: rightAngle ?? null,
        leftKneeScore: leftScore ?? 0,
        rightKneeScore: rightScore ?? 0
      };
      try { await fetch('/pose', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) }); }
      catch(e){ /* ignore for now */ }
    }
    
    function angleAt(center, a, c) {
      // Calculate angle at center point between vectors center->a and center->c
      const ax = a.x - center.x;
      const ay = a.y - center.y;
      const cx = c.x - center.x;
      const cy = c.y - center.y;
      
      const dotProduct = ax * cx + ay * cy;
      const magA = Math.sqrt(ax * ax + ay * ay);
      const magC = Math.sqrt(cx * cx + cy * cy);
      
      if (magA === 0 || magC === 0) return 0;
      
      const cosAngle = dotProduct / (magA * magC);
      const angleRad = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
      return angleRad * (180 / Math.PI);
    }
    
    window.__start = async () => {
      const s=document.getElementById('status'); const v=document.getElementById('video');
      try {
        s.textContent='Requesting camera…';
        const stream = await navigator.mediaDevices.getUserMedia({ video:{width:640,height:480,facingMode:'user'}, audio:false });
        v.srcObject = stream;
        v.onloadedmetadata = async () => { 
          s.textContent='Camera ready, loading AI model...';
          
          // Initialize TensorFlow and MoveNet
          await tf.ready();
          const detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
          };
          const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
          
          s.textContent='AI model loaded, tracking poses...';
          
          // Start pose detection loop
          const detectPoses = async () => {
            try {
              const poses = await detector.estimatePoses(v);
              
              const c = document.getElementById('overlay'); const ctx = c.getContext('2d');
              const vw = v.videoWidth || 640, vh = v.videoHeight || 480;
              c.width = v.clientWidth || 640;
              c.height = v.clientHeight || 480;
              const sx = c.width / vw, sy = c.height / vh;
              ctx.clearRect(0,0,c.width,c.height); ctx.lineWidth=3; ctx.lineCap='round'; ctx.lineJoin='round';
              
              if (poses.length > 0) {
                const keypoints = poses[0].keypoints;
                
                // MoveNet keypoint indices
                const LEFT_HIP = 11, LEFT_KNEE = 13, LEFT_ANKLE = 15;
                const RIGHT_HIP = 12, RIGHT_KNEE = 14, RIGHT_ANKLE = 16;
                
                const lHip = keypoints[LEFT_HIP];
                const lKnee = keypoints[LEFT_KNEE];
                const lAnkle = keypoints[LEFT_ANKLE];
                
                const rHip = keypoints[RIGHT_HIP];
                const rKnee = keypoints[RIGHT_KNEE];
                const rAnkle = keypoints[RIGHT_ANKLE];
                
                let leftAngle=null, rightAngle=null, leftScore=0, rightScore=0;
                
                // Draw left leg overlay and update angle
                if (lHip.score >= 0.4 && lKnee.score >= 0.4 && lAnkle.score >= 0.4) {
                  leftScore = Math.min(lHip.score, lKnee.score, lAnkle.score);
                  leftAngle = angleAt(lKnee, lHip, lAnkle);
                  document.getElementById('lAng').textContent = leftAngle.toFixed(1);
                  
                   // Draw left leg lines
                   ctx.strokeStyle = '#FF6B6B';
                   ctx.beginPath();
                   ctx.moveTo(lHip.x * sx, lHip.y * sy);
                   ctx.lineTo(lKnee.x * sx, lKnee.y * sy);
                   ctx.lineTo(lAnkle.x * sx, lAnkle.y * sy);
                   ctx.stroke();
                   
                   // Draw left leg joint circles
                   ctx.fillStyle = '#FF6B6B';
                   [lHip, lKnee, lAnkle].forEach(joint => {
                     ctx.beginPath();
                     ctx.arc(joint.x * sx, joint.y * sy, 6, 0, 2 * Math.PI);
                     ctx.fill();
                     
                     // White inner dot
                     ctx.fillStyle = 'white';
                     ctx.beginPath();
                     ctx.arc(joint.x * sx, joint.y * sy, 2, 0, 2 * Math.PI);
                     ctx.fill();
                     ctx.fillStyle = '#FF6B6B';
                   });
                } else {
                  document.getElementById('lAng').textContent = '--';
                }
                
                // Draw right leg overlay and update angle
                if (rHip.score >= 0.4 && rKnee.score >= 0.4 && rAnkle.score >= 0.4) {
                  rightScore = Math.min(rHip.score, rKnee.score, rAnkle.score);
                  rightAngle = angleAt(rKnee, rHip, rAnkle);
                  document.getElementById('rAng').textContent = rightAngle.toFixed(1);
                  
                   // Draw right leg lines
                   ctx.strokeStyle = '#4ECDC4';
                   ctx.beginPath();
                   ctx.moveTo(rHip.x * sx, rHip.y * sy);
                   ctx.lineTo(rKnee.x * sx, rKnee.y * sy);
                   ctx.lineTo(rAnkle.x * sx, rAnkle.y * sy);
                   ctx.stroke();
                   
                   // Draw right leg joint circles
                   ctx.fillStyle = '#4ECDC4';
                   [rHip, rKnee, rAnkle].forEach(joint => {
                     ctx.beginPath();
                     ctx.arc(joint.x * sx, joint.y * sy, 6, 0, 2 * Math.PI);
                     ctx.fill();
                     
                     // White inner dot
                     ctx.fillStyle = 'white';
                     ctx.beginPath();
                     ctx.arc(joint.x * sx, joint.y * sy, 2, 0, 2 * Math.PI);
                     ctx.fill();
                     ctx.fillStyle = '#4ECDC4';
                   });
                } else {
                  document.getElementById('rAng').textContent = '--';
                }
                
                sendAngles(leftAngle, rightAngle, leftScore, rightScore);
              }
            } catch (err) {
              console.error('Pose detection error:', err);
            }
            
            requestAnimationFrame(detectPoses);
          };
          
          detectPoses();
        };
      } catch(err){ s.textContent='Error: '+(err.message||err); }
    };
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/pose")
async def pose(request: Request):
    global latest
    latest = await request.json()
    return {"status": "received"}

@app.get("/pose/latest")
async def pose_latest():
    return latest if latest else {}

if __name__ == "__main__":
    print("Open: http://127.0.0.1:8081")
    webbrowser.open("http://127.0.0.1:8081")
    uvicorn.run(app, host="127.0.0.1", port=8081)