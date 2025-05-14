import cv2
from cv2_enumerate_cameras import enumerate_cameras
import numpy as np
from flask import Flask, Response, send_file, jsonify
import threading
from io import BytesIO

"""
Webcam Flask server with named camera selection and backward‑compat /single_frame alias.
Run with:
    pip install opencv-python cv2-enumerate-cameras Flask numpy
"""

app = Flask(__name__)

# ------------------------- Globals -------------------------
cap = None
frame_lock = threading.Lock()
current_camera_index = 0
available_cameras = {}

# -------------------- Camera enumeration -------------------

def scan_available_cameras():
    """Populate ``available_cameras`` using DirectShow enumeration."""
    global available_cameras
    available_cameras = {info.index: info.name for info in enumerate_cameras(cv2.CAP_DSHOW)}
    if not available_cameras:
        print("No cameras found via DirectShow.")
    else:
        print("Available cameras:")
        for idx, name in available_cameras.items():
            print(f"  {idx}: {name}")
    return available_cameras


def get_camera_name(index: int) -> str:
    return available_cameras.get(index, f"Camera {index}")


# ----------------------- Camera open -----------------------

def open_camera(index: int):
    """Open camera by index via DirectShow backend."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cap if cap.isOpened() else None


def get_webcam(index: int):
    """Return an opened ``cv2.VideoCapture`` for *index* (switching if needed)."""
    global cap, current_camera_index

    # If already open and requested index matches, reuse
    if cap is not None and cap.isOpened() and current_camera_index == index:
        return cap

    # Close current cam if mismatch / closed
    if cap is not None:
        cap.release()

    cap = open_camera(index)
    if cap is not None:
        current_camera_index = index
        print(f"Opened {get_camera_name(index)} (index {index})")
    else:
        print(f"Failed to open camera {index}")

    return cap


def init_webcam() -> bool:
    """Ensure at least one camera is open; return True on success."""
    if not available_cameras:
        scan_available_cameras()

    if current_camera_index not in available_cameras:
        # pick first available
        if not available_cameras:
            return False
        first = sorted(available_cameras)[0]
        print(f"Defaulting to camera {first}")
        return get_webcam(first) is not None

    return get_webcam(current_camera_index) is not None


# ----------------------- Flask routes ----------------------

@app.route('/')
def index():
    """UI for changing which webcam as needed."""
    return (
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Webcam Server</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
                #videoFeed { width: 100%; border: 1px solid #ddd; }
                .controls { margin-top: 15px; }
                button { padding: 6px 14px; margin: 4px; cursor: pointer; }
                .camera-button.active { background:#4CAF50; color:#fff; }
            </style>
        </head>
        <body>
            <h1>Webcam Server</h1>
            <img id="videoFeed" src="/video_feed" />

            <div class="controls">
                <p>Current camera: <strong id="currentCamera">Loading…</strong></p>
                <button onclick="switchCamera(-1)">Prev</button>
                <button onclick="switchCamera(1)">Next</button>
                <button onclick="scanCams()">Rescan</button>
                <div id="camButtons"></div>
            </div>

            <script>
                let cams = {}; let current = 0;
                window.onload = refresh;

                function refresh() {
                    fetch('/camera_info').then(r=>r.json()).then(data=>{
                        cams = data.available; current = data.index;
                        document.getElementById('currentCamera').textContent = data.name + ' ('+current+')';
                        redraw();
                    });
                }
                function scanCams() { fetch('/scan').then(()=>refresh()); }
                function switchCamera(dir){
                    const keys = Object.keys(cams).map(Number).sort((a,b)=>a-b);
                    if(!keys.length) return;
                    let i = keys.indexOf(current)+dir;
                    if(i<0) i=keys.length-1; if(i>=keys.length) i=0;
                    setCam(keys[i]);
                }
                function setCam(idx){ fetch('/set/'+idx).then(r=>r.json()).then(res=>{ if(res.success) refresh(); else alert(res.error);}); }
                function redraw(){
                    const div=document.getElementById('camButtons'); div.innerHTML='';
                    Object.entries(cams).sort((a,b)=>a[0]-b[0]).forEach(([i,name])=>{
                        const b=document.createElement('button'); b.textContent=`${name} (${i})`; b.onclick=()=>setCam(i);
                        if(Number(i)===current) b.className='camera-button active'; div.appendChild(b);
                    });
                }
            </script>
        </body>
        </html>
        """
    )


@app.route('/camera_info')
def camera_info():
    if not available_cameras:
        scan_available_cameras()
    return jsonify({
        "index": current_camera_index,
        "name": get_camera_name(current_camera_index),
        "available": available_cameras
    })


@app.route('/scan')
def scan():
    return jsonify({"available": scan_available_cameras()})


@app.route('/set/<int:index>')
def set_camera(index):
    with frame_lock:
        if index not in available_cameras:
            return jsonify({"success": False, "error": "Index not recognised"})
        if get_webcam(index):
            return jsonify({"success": True})
        return jsonify({"success": False, "error": f"Camera {index} failed to open"})


# ---------------- single frame routes ----------------------

@app.route('/single')
def single_frame():
    with frame_lock:
        if not init_webcam():
            return "No camera", 500
        ret, frame = cap.read()
        if not ret:
            return "Capture failed", 500
        _, buf = cv2.imencode('.jpg', frame)
        return send_file(BytesIO(buf), mimetype='image/jpeg')

# Legacy alias for existing client scripts
@app.route('/single_frame')
def single_frame_alias():
    return single_frame()

# ---------------- video stream -----------------------------

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock:
                if not init_webcam():
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------- Main entry -------------------------

if __name__ == '__main__':
    scan_available_cameras()
    init_webcam()
    app.run(host='0.0.0.0', port=5000, threaded=True)
