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
Knee (abs, 0–1 screen): L(0.000, 0.000)   R(0.000, 0.000)
Foot (abs, 0–1 screen): L(0.000, 0.000)   R(0.000, 0.000)
  </pre>

  <div id="stack" style="position:relative;width:640px;height:480px;">
    <video id="video" width="640" height="480" autoplay muted playsinline style="display:block;position:absolute;left:0;top:0;z-index:1;"></video>
    <canvas id="overlay" width="640" height="480" style="position:absolute;left:0;top:0;pointer-events:none;z-index:2;"></canvas>
  </div>

  <script>
    window.addEventListener('error', e => {
      const s=document.getElementById('status');
      if(s) s.textContent='JS error: '+e.message;
    });

    function clamp01(t){ return Math.max(0, Math.min(1, t)); }

    // Robust angle filter knobs
    const MED_N = 5;     // median window (last N valid frames)
    const MAX_STEP = 12; // max degrees/frame allowed (anti-spike)
    const GAP_HOLD = 6;  // tolerate up to 6 invalid frames before null

    // Filter state
    let lBuf = [], rBuf = [];
    let lOut = null, rOut = null;
    let lGap = 0,   rGap = 0;

    let sendFrame = 0, lastPost = 0;
    const SEND_MS = 50; // ~20 Hz
    const isMirroredPreview = false; // UI-only

    async function sendAngles(leftAngle, rightAngle, leftScore, rightScore, hipData){
      const now = Date.now();
      if (now - lastPost < SEND_MS) return;
      lastPost = now;

      const payload = {
        schema: 3,
        frameId: sendFrame++,
        ts: now/1000,

        leftKneeAngle: leftAngle ?? null,
        rightKneeAngle: rightAngle ?? null,
        leftKneeScore: leftScore ?? 0,
        rightKneeScore: rightScore ?? 0,

        // hips (0–1 screen)
        hipValid: hipData.hipValid,
        hipL_x: hipData.hipL_x, hipL_y: hipData.hipL_y,
        hipR_x: hipData.hipR_x, hipR_y: hipData.hipR_y,
        hipC_x: hipData.hipC_x, hipC_y: hipData.hipC_y,

        // knees/feet (absolute 0–1 screen)
        kneeL_x: hipData.kneeL_x, kneeL_y: hipData.kneeL_y,
        kneeR_x: hipData.kneeR_x, kneeR_y: hipData.kneeR_y,
        footL_x: hipData.footL_x, footL_y: hipData.footL_y,
        footR_x: hipData.footR_x, footR_y: hipData.footR_y
      };

      if ((sendFrame % 20) === 0) console.log('pose payload', payload);

      try {
        await fetch('/pose', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(payload)
        });
      } catch(e){ /* ignore for now */ }
    }

    // draw preview mirrored, math stays raw
    function drawMirrored(ctx, w, h, drawFn) {
      ctx.save();
      if (isMirroredPreview) {
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
      }
      drawFn();
      ctx.restore();
    }

    function angleAt(center, a, c) {
      // angle at "center" between vectors center->a and center->c
      const ax = a.x - center.x, ay = a.y - center.y;
      const cx = c.x - center.x, cy = c.y - center.y;
      const dot = ax*cx + ay*cy;
      const magA = Math.sqrt(ax*ax + ay*ay);
      const magC = Math.sqrt(cx*cx + cy*cy);
      if (!magA || !magC) return 0;
      const cos = dot / (magA * magC);
      const ang = Math.acos(Math.max(-1, Math.min(1, cos)));
      return ang * 180 / Math.PI;
    }

    window.__start = async () => {
      const s=document.getElementById('status');
      const v=document.getElementById('video');

      try {
        s.textContent='Requesting camera…';
        const stream = await navigator.mediaDevices.getUserMedia({
          video:{width:640,height:480,facingMode:'user'},
          audio:false
        });
        v.srcObject = stream;

        v.onloadedmetadata = async () => {
          s.textContent='Camera ready, loading AI model...';

          await tf.ready();
          const detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
          };
          const detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            detectorConfig
          );

          s.textContent='AI model loaded, tracking poses...';

          const detectPoses = async () => {
            try {
              // IMPORTANT: math in raw camera space (no flip)
              const poses = await detector.estimatePoses(v, { flipHorizontal: false });

              const c = document.getElementById('overlay');
              const ctx = c.getContext('2d');
              const vw = v.videoWidth || 640, vh = v.videoHeight || 480;
              c.width = v.clientWidth || 640;
              c.height = v.clientHeight || 480;
              const sx = c.width / vw, sy = c.height / vh;

              ctx.clearRect(0,0,c.width,c.height);
              ctx.lineWidth=3; ctx.lineCap='round'; ctx.lineJoin='round';

              const CONF_T = 0.4;

              if (poses.length > 0) {
                const kps = poses[0].keypoints;

                // indices
                const LEFT_HIP = 11, LEFT_KNEE = 13, LEFT_ANKLE = 15;
                const RIGHT_HIP = 12, RIGHT_KNEE = 14, RIGHT_ANKLE = 16;

                const lHip = kps[LEFT_HIP],  lKnee = kps[LEFT_KNEE],  lAnkle = kps[LEFT_ANKLE];
                const rHip = kps[RIGHT_HIP], rKnee = kps[RIGHT_KNEE], rAnkle = kps[RIGHT_ANKLE];

                // absolute 0–1 screen coords (clamped)
                const kneeL_x = clamp01(lKnee.x / vw), kneeL_y = clamp01(lKnee.y / vh);
                const kneeR_x = clamp01(rKnee.x / vw), kneeR_y = clamp01(rKnee.y / vh);
                const footL_x = clamp01(lAnkle.x / vw), footL_y = clamp01(lAnkle.y / vh);
                const footR_x = clamp01(rAnkle.x / vw), footR_y = clamp01(rAnkle.y / vh);

                let leftAngle=null, rightAngle=null, leftScore=0, rightScore=0;

                // hip center + hip normalized (0–1)
                const haveLHip = lHip.score >= CONF_T;
                const haveRHip = rHip.score >= CONF_T;
                const hipValid = haveLHip || haveRHip;

                let hipC = {x:0,y:0}, pelvisWidth = 1;
                let hipL_x = 0, hipL_y = 0, hipR_x = 0, hipR_y = 0, hipC_x = 0, hipC_y = 0;

                if (hipValid) {
                  if (haveLHip && haveRHip) {
                    hipC.x = (lHip.x + rHip.x) / 2;
                    hipC.y = (lHip.y + rHip.y) / 2;
                    pelvisWidth = Math.max(1, Math.hypot(lHip.x - rHip.x, lHip.y - rHip.y));
                  } else if (haveLHip) {
                    hipC.x = lHip.x; hipC.y = lHip.y;
                    if (lKnee.score >= CONF_T) pelvisWidth = Math.max(1, Math.hypot(lHip.x - lKnee.x, lHip.y - lKnee.y));
                  } else {
                    hipC.x = rHip.x; hipC.y = rHip.y;
                    if (rKnee.score >= CONF_T) pelvisWidth = Math.max(1, Math.hypot(rHip.x - rKnee.x, rHip.y - rKnee.y));
                  }

                  hipL_x = clamp01(lHip.x / vw); hipL_y = clamp01(lHip.y / vh);
                  hipR_x = clamp01(rHip.x / vw); hipR_y = clamp01(rHip.y / vh);
                  hipC_x = clamp01(hipC.x / vw); hipC_y = clamp01(hipC.y / vh);
                }

                // angles
                if (lHip.score >= CONF_T && lKnee.score >= CONF_T && lAnkle.score >= CONF_T) {
                  leftScore = Math.min(lHip.score, lKnee.score, lAnkle.score);
                  leftAngle = angleAt(lKnee, lHip, lAnkle);
                }
                if (rHip.score >= CONF_T && rKnee.score >= CONF_T && rAnkle.score >= CONF_T) {
                  rightScore = Math.min(rHip.score, rKnee.score, rAnkle.score);
                  rightAngle = angleAt(rKnee, rHip, rAnkle);
                }

                // draw overlay (optionally mirrored for display only)
                drawMirrored(ctx, c.width, c.height, () => {
                  if (lHip.score >= CONF_T && lKnee.score >= CONF_T && lAnkle.score >= CONF_T) {
                    ctx.strokeStyle = '#FF6B6B';
                    ctx.beginPath();
                    ctx.moveTo(lHip.x * sx, lHip.y * sy);
                    ctx.lineTo(lKnee.x * sx, lKnee.y * sy);
                    ctx.lineTo(lAnkle.x * sx, lAnkle.y * sy);
                    ctx.stroke();

                    ctx.fillStyle = '#FF6B6B';
                    [lHip, lKnee, lAnkle].forEach(j => {
                      ctx.beginPath(); ctx.arc(j.x * sx, j.y * sy, 6, 0, 2*Math.PI); ctx.fill();
                      ctx.fillStyle = 'white';
                      ctx.beginPath(); ctx.arc(j.x * sx, j.y * sy, 2, 0, 2*Math.PI); ctx.fill();
                      ctx.fillStyle = '#FF6B6B';
                    });
                  }

                  if (rHip.score >= CONF_T && rKnee.score >= CONF_T && rAnkle.score >= CONF_T) {
                    ctx.strokeStyle = '#4ECDC4';
                    ctx.beginPath();
                    ctx.moveTo(rHip.x * sx, rHip.y * sy);
                    ctx.lineTo(rKnee.x * sx, rKnee.y * sy);
                    ctx.lineTo(rAnkle.x * sx, rAnkle.y * sy);
                    ctx.stroke();

                    ctx.fillStyle = '#4ECDC4';
                    [rHip, rKnee, rAnkle].forEach(j => {
                      ctx.beginPath(); ctx.arc(j.x * sx, j.y * sy, 6, 0, 2*Math.PI); ctx.fill();
                      ctx.fillStyle = 'white';
                      ctx.beginPath(); ctx.arc(j.x * sx, j.y * sy, 2, 0, 2*Math.PI); ctx.fill();
                      ctx.fillStyle = '#4ECDC4';
                    });
                  }

                  if (hipValid) {
                    ctx.strokeStyle = '#FFD700'; ctx.lineWidth = 2;
                    const hx = hipC_x * vw * sx, hy = hipC_y * vh * sy;
                    ctx.beginPath(); ctx.moveTo(hx-8,hy); ctx.lineTo(hx+8,hy); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(hx,hy-8); ctx.lineTo(hx,hy+8); ctx.stroke();
                  }
                });

                const hipData = {
                  hipValid,
                  hipL_x, hipL_y, hipR_x, hipR_y, hipC_x, hipC_y,
                  // absolute 0–1 screen coords
                  kneeL_x, kneeL_y, kneeR_x, kneeR_y,
                  footL_x, footL_y, footR_x, footR_y
                };

                // Filter angles before UI/send
                const fmt3 = v => Number.isFinite(v) ? v.toFixed(3) : '0.000';
                const leftValid  = leftAngle  != null && leftScore  >= CONF_T;
                const rightValid = rightAngle != null && rightScore >= CONF_T;

                function median(arr){
                  const a=[...arr].sort((x,y)=>x-y);
                  const m=Math.floor(a.length/2);
                  return a.length%2 ? a[m] : (a[m-1]+a[m])/2;
                }

                // LEFT
                if (leftValid && Number.isFinite(leftAngle) && leftAngle>=0 && leftAngle<=180) {
                  if (lOut != null) {
                    const delta = leftAngle - lOut;
                    if (Math.abs(delta) > MAX_STEP) leftAngle = lOut + Math.sign(delta) * MAX_STEP;
                  }
                  lBuf.push(leftAngle);
                  if (lBuf.length > MED_N) lBuf.shift();
                  lOut = median(lBuf);
                  lGap = 0;
                } else {
                  lGap++;
                  if (lGap > GAP_HOLD) { lOut = null; lBuf.length = 0; }
                }

                // RIGHT
                if (rightValid && Number.isFinite(rightAngle) && rightAngle>=0 && rightAngle<=180) {
                  if (rOut != null) {
                    const delta = rightAngle - rOut;
                    if (Math.abs(delta) > MAX_STEP) rightAngle = rOut + Math.sign(delta) * MAX_STEP;
                  }
                  rBuf.push(rightAngle);
                  if (rBuf.length > MED_N) rBuf.shift();
                  rOut = median(rBuf);
                  rGap = 0;
                } else {
                  rGap++;
                  if (rGap > GAP_HOLD) { rOut = null; rBuf.length = 0; }
                }

                const readout = `
L: ${lOut != null ? lOut.toFixed(1) : '--'}° | R: ${rOut != null ? rOut.toFixed(1) : '--'}°
Hips (normalized): L(${fmt3(hipData.hipL_x)}, ${fmt3(hipData.hipL_y)}) R(${fmt3(hipData.hipR_x)}, ${fmt3(hipData.hipR_y)}) C(${fmt3(hipData.hipC_x)}, ${fmt3(hipData.hipC_y)}) [${hipData.hipValid ? 'OK' : '--'}]
Knee (abs, 0–1 screen): L(${fmt3(hipData.kneeL_x)}, ${fmt3(hipData.kneeL_y)})   R(${fmt3(hipData.kneeR_x)}, ${fmt3(hipData.kneeR_y)})
Foot (abs, 0–1 screen): L(${fmt3(hipData.footL_x)}, ${fmt3(hipData.footL_y)})   R(${fmt3(hipData.footR_x)}, ${fmt3(hipData.footR_y)})`;
                document.getElementById('readout').textContent = readout;

                sendAngles(lOut ?? leftAngle, rOut ?? rightAngle, leftScore, rightScore, hipData);
              }
            } catch (err) {
              console.error('Pose detection error:', err);
            }

            requestAnimationFrame(detectPoses);
          };

          detectPoses();
        };
      } catch(err){
        s.textContent='Error: '+(err.message||err);
      }
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
        "l_deg":      latest_full.get("leftKneeAngle"),
        "r_deg":      latest_full.get("rightKneeAngle"),

        "hip_ok":     latest_full.get("hipValid", False),
        "hipL_x":     latest_full.get("hipL_x", 0.0),
        "hipL_y":     latest_full.get("hipL_y", 0.0),
        "hipR_x":     latest_full.get("hipR_x", 0.0),
        "hipR_y":     latest_full.get("hipR_y", 0.0),
        "hipC_x":     latest_full.get("hipC_x", 0.0),
        "hipC_y":     latest_full.get("hipC_y", 0.0),

        # absolute, screen-normalized (0–1)
        "kneeL_x":    latest_full.get("kneeL_x", 0.0),
        "kneeL_y":    latest_full.get("kneeL_y", 0.0),
        "kneeR_x":    latest_full.get("kneeR_x", 0.0),
        "kneeR_y":    latest_full.get("kneeR_y", 0.0),
        "footL_x":    latest_full.get("footL_x", 0.0),
        "footL_y":    latest_full.get("footL_y", 0.0),
        "footR_x":    latest_full.get("footR_x", 0.0),
        "footR_y":    latest_full.get("footR_y", 0.0),
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

