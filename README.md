\# ACL Pose Relay → Unreal (VaRest)



Single-file app (`app.py`) that:

\- Opens a browser \*\*debug UI\*\* (webcam + MoveNet) and shows \*\*left/right knee angles\*\* with an overlay.

\- Exposes HTTP endpoints so \*\*Unreal (VaRest)\*\* can poll and print the values.



\*\*Default URL:\*\* `http://127.0.0.1:8081`



---



\## Requirements

\- Python \*\*3.10+\*\*

\- Google \*\*Chrome/Edge\*\* (for webcam)

\- Unreal Engine with \*\*VaRest\*\* plugin



---



\## Setup \& Run (local)



```bash

\# from the repo folder

python -m venv .venv

\# Windows

.\\.venv\\Scripts\\activate

\# macOS/Linux

source .venv/bin/activate



pip install fastapi uvicorn

python app.py



