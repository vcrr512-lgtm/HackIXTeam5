from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
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

# Global variables to store latest pose data
latest_full = {}
latest_view = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Camera Smoke Test</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.0/dist/pose-detection.min.js"></script>
    <style>
      body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color:#222; background:#fff; }
      #metrics {
        margin-top:10px; font-size:14px; line-height:1.4;
        background:#f7f7f9; color:#222;
        border:1px solid #e5e7eb; border-radius:8px; padding:10px 12px;
        max-width:640px; box-shadow:0 1px 2px rgba(0,0,0,.05);
      }
      #metrics b { color:#111; }
    </style>
</head>
<body>
    <h1>Camera Smoke Test</h1>
    <div id="status" style="color: green;">Ready</div>
    <button id="startBtn" onclick="window.__start()">Start Camera</button>
    <pre id="readout" style="margin-top:10px;font:14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;color:#222;background:transparent;border:none;padding:0;">
L: --° | R: --°
Hips (normalized): L(0.000, 0.000) R(0.000, 0.000) C(0.000, 0.000) [--]
Knee Δ (rel to hipC, /pelvisWidth): L(0.000, 0.000) [--]   R(0.000, 0.000) [--]
Foot Δ (rel to hipC, /pelvisWidth): L(0.000, 0.000) [--]   R(0.000, 0.000) [--]
</pre>
    <br><br>
    <div id="stack" style="position:relative;width:640px;height:480px;">
      <video id="video" width="640" height="480" autoplay muted playsinline style="display:block;position:absolute;left:0;top:0;z-index:1;"></video>
      <canvas id="overlay" width="640" height="480" style="position:absolute;left:0;top:0;pointer-events:none;z-index:2;"></canvas>
    </div>
    
    <script>
    window.addEventListener('error', e => { const s=document.getElementById('status'); if(s) s.textContent='JS error: '+e.message; });
    
    let sendFrame = 0, lastPost = 0;
    const SEND_MS = 50; // ~20 Hz
    
    async function sendAngles(leftAngle, rightAngle, leftScore, rightScore, hipData){
      const now = Date.now();
      if (now - lastPost < SEND_MS) return;
      lastPost = now;
      const payload = {
        schema: 2,
        frameId: sendFrame++,
        ts: now/1000,
        leftKneeAngle: leftAngle ?? null,
        rightKneeAngle: rightAngle ?? null,
        leftKneeScore: leftScore ?? 0,
        rightKneeScore: rightScore ?? 0,
        // Hip absolute positions
        hipValid: hipData.hipValid,
        hipL_x: hipData.hipL_x,
        hipL_y: hipData.hipL_y,
        hipR_x: hipData.hipR_x,
        hipR_y: hipData.hipR_y,
        hipC_x: hipData.hipC_x,
        hipC_y: hipData.hipC_y,
        // Knee relative offsets
        kneeL_valid: hipData.kneeL_valid,
        kneeL_dx: hipData.kneeL_dx,
        kneeL_dy: hipData.kneeL_dy,
        kneeR_valid: hipData.kneeR_valid,
        kneeR_dx: hipData.kneeR_dx,
        kneeR_dy: hipData.kneeR_dy,
        // Foot relative offsets
        footL_valid: hipData.footL_valid,
        footL_dx: hipData.footL_dx,
        footL_dy: hipData.footL_dy,
        footR_valid: hipData.footR_valid,
        footR_dx: hipData.footR_dx,
        footR_dy: hipData.footR_dy
      };
      if ((sendFrame % 20) === 0) console.log('pose payload', payload);
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
              
              const CONF_T = 0.4;
              
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
                
                // Relaxed hip gating with fallbacks
                const haveLHip = lHip.score >= CONF_T;
                const haveRHip = rHip.score >= CONF_T;
                const hipValid = haveLHip || haveRHip; // either hip is enough
                
                let hipC = {x: 0, y: 0}, pelvisWidth = 1;
                let hipL_x = 0, hipL_y = 0, hipR_x = 0, hipR_y = 0, hipC_x = 0, hipC_y = 0;
                let kneeL_valid = false, kneeL_dx = 0, kneeL_dy = 0;
                let kneeR_valid = false, kneeR_dx = 0, kneeR_dy = 0;
                let footL_valid = false, footL_dx = 0, footL_dy = 0;
                let footR_valid = false, footR_dx = 0, footR_dy = 0;
                
                if (hipValid) {
                  // Compute hip center
                  if (haveLHip && haveRHip) {
                    hipC.x = (lHip.x + rHip.x) / 2;
                    hipC.y = (lHip.y + rHip.y) / 2;
                  } else if (haveLHip) {
                    hipC.x = lHip.x;
                    hipC.y = lHip.y;
                  } else {
                    hipC.x = rHip.x;
                    hipC.y = rHip.y;
                  }
                  
                  // Compute pelvis width
                  if (haveLHip && haveRHip) {
                    pelvisWidth = Math.max(1, Math.sqrt((lHip.x - rHip.x)**2 + (lHip.y - rHip.y)**2));
                  } else if (haveLHip && lKnee.score >= CONF_T) {
                    pelvisWidth = Math.max(1, Math.sqrt((lHip.x - lKnee.x)**2 + (lHip.y - lKnee.y)**2));
                  } else if (haveRHip && rKnee.score >= CONF_T) {
                    pelvisWidth = Math.max(1, Math.sqrt((rHip.x - rKnee.x)**2 + (rHip.y - rKnee.y)**2));
                  }
                  
                  // Absolute normalized hip positions
                  hipL_x = lHip.x / vw;
                  hipL_y = lHip.y / vh;
                  hipR_x = rHip.x / vw;
                  hipR_y = rHip.y / vh;
                  hipC_x = hipC.x / vw;
                  hipC_y = hipC.y / vh;
                  
                  // Relative normalized knee offsets
                  kneeL_valid = (lKnee.score >= CONF_T);
                  if (kneeL_valid) {
                    kneeL_dx = (lKnee.x - hipC.x) / pelvisWidth;
                    kneeL_dy = (lKnee.y - hipC.y) / pelvisWidth;
                  }
                  
                  kneeR_valid = (rKnee.score >= CONF_T);
                  if (kneeR_valid) {
                    kneeR_dx = (rKnee.x - hipC.x) / pelvisWidth;
                    kneeR_dy = (rKnee.y - hipC.y) / pelvisWidth;
                  }
                  
                  // Relative normalized foot offsets
                  footL_valid = (lAnkle.score >= CONF_T);
                  if (footL_valid) {
                    footL_dx = (lAnkle.x - hipC.x) / pelvisWidth;
                    footL_dy = (lAnkle.y - hipC.y) / pelvisWidth;
                  }
                  
                  footR_valid = (rAnkle.score >= CONF_T);
                  if (footR_valid) {
                    footR_dx = (rAnkle.x - hipC.x) / pelvisWidth;
                    footR_dy = (rAnkle.y - hipC.y) / pelvisWidth;
                  }
                }
                
                // Draw left leg overlay and update angle
                if (lHip.score >= 0.4 && lKnee.score >= 0.4 && lAnkle.score >= 0.4) {
                  leftScore = Math.min(lHip.score, lKnee.score, lAnkle.score);
                  leftAngle = angleAt(lKnee, lHip, lAnkle);
                  
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
                }
                
                // Draw right leg overlay and update angle
                if (rHip.score >= 0.4 && rKnee.score >= 0.4 && rAnkle.score >= 0.4) {
                  rightScore = Math.min(rHip.score, rKnee.score, rAnkle.score);
                  rightAngle = angleAt(rKnee, rHip, rAnkle);
                  
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
                }
                
                const hipData = {
                  hipValid, hipL_x, hipL_y, hipR_x, hipR_y, hipC_x, hipC_y,
                  kneeL_valid, kneeL_dx, kneeL_dy, kneeR_valid, kneeR_dx, kneeR_dy,
                  footL_valid, footL_dx, footL_dy, footR_valid, footR_dx, footR_dy
                };
                
                // Update readout
                const fmt3 = v => Number.isFinite(v) ? v.toFixed(3) : '0.000';
                const leftValid  = leftAngle  != null && leftScore  >= 0.4;
                const rightValid = rightAngle != null && rightScore >= 0.4;
                const readout =
                  `L: ${leftValid ? leftAngle.toFixed(1) : '--'}° | R: ${rightValid ? rightAngle.toFixed(1) : '--'}°\n` +
                  `Hips (normalized): L(${fmt3(hipData.hipL_x)}, ${fmt3(hipData.hipL_y)}) ` +
                  `R(${fmt3(hipData.hipR_x)}, ${fmt3(hipData.hipR_y)}) ` +
                  `C(${fmt3(hipData.hipC_x)}, ${fmt3(hipData.hipC_y)}) ` +
                  `[${hipData.hipValid ? 'OK' : '--'}]\n` +
                  `Knee Δ (rel to hipC, /pelvisWidth): ` +
                  `L(${fmt3(hipData.kneeL_dx)}, ${fmt3(hipData.kneeL_dy)}) [${hipData.kneeL_valid ? 'OK' : '--'}]   ` +
                  `R(${fmt3(hipData.kneeR_dx)}, ${fmt3(hipData.kneeR_dy)}) [${hipData.kneeR_valid ? 'OK' : '--'}]\n` +
                  `Foot Δ (rel to hipC, /pelvisWidth): ` +
                  `L(${fmt3(hipData.footL_dx)}, ${fmt3(hipData.footL_dy)}) [${hipData.footL_valid ? 'OK' : '--'}]   ` +
                  `R(${fmt3(hipData.footR_dx)}, ${fmt3(hipData.footR_dy)}) [${hipData.footR_valid ? 'OK' : '--'}]`;
                document.getElementById('readout').textContent = readout;
                
                // Draw hip center marker for visual sanity
                if (hipData.hipValid) {
                  ctx.strokeStyle = '#FFD700'; ctx.lineWidth = 2;
                  const hx = hipData.hipC_x * vw * sx, hy = hipData.hipC_y * vh * sy;
                  ctx.beginPath(); ctx.moveTo(hx-8,hy); ctx.lineTo(hx+8,hy); ctx.stroke();
                  ctx.beginPath(); ctx.moveTo(hx,hy-8); ctx.lineTo(hx,hy+8); ctx.stroke();
                }
                
                sendAngles(leftAngle, rightAngle, leftScore, rightScore, hipData);
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
    global latest_full, latest_view
    latest_full = await request.json()
    
    # Build slim view with only essential fields
    latest_view = {
        "l_deg": latest_full.get("leftKneeAngle"),
        "r_deg": latest_full.get("rightKneeAngle"),
        "hipL_x": latest_full.get("hipL_x", 0),
        "hipL_y": latest_full.get("hipL_y", 0),
        "hipR_x": latest_full.get("hipR_x", 0),
        "hipR_y": latest_full.get("hipR_y", 0),
        "hipC_x": latest_full.get("hipC_x", 0),
        "hipC_y": latest_full.get("hipC_y", 0),
        "hip_ok": latest_full.get("hipValid", False),
        "kneeL_dx": latest_full.get("kneeL_dx", 0),
        "kneeL_dy": latest_full.get("kneeL_dy", 0),
        "kneeL_ok": latest_full.get("kneeL_valid", False),
        "kneeR_dx": latest_full.get("kneeR_dx", 0),
        "kneeR_dy": latest_full.get("kneeR_dy", 0),
        "kneeR_ok": latest_full.get("kneeR_valid", False),
        "footL_dx": latest_full.get("footL_dx", 0),
        "footL_dy": latest_full.get("footL_dy", 0),
        "footL_ok": latest_full.get("footL_valid", False),
        "footR_dx": latest_full.get("footR_dx", 0),
        "footR_dy": latest_full.get("footR_dy", 0),
        "footR_ok": latest_full.get("footR_valid", False)
    }
    
    return {"status": "received"}

@app.get("/pose/latest")
async def pose_latest():
    content = latest_view if latest_view else {}
    return JSONResponse(content=content, headers={"Cache-Control": "no-store"})

@app.get("/pose/full")
async def pose_full():
    content = latest_full if latest_full else {}
    return JSONResponse(content=content, headers={"Cache-Control": "no-store"})

if __name__ == "__main__":
    print("Open: http://127.0.0.1:8081")
    webbrowser.open("http://127.0.0.1:8081")
    uvicorn.run(app, host="127.0.0.1", port=8081)