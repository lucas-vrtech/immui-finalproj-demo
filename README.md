# immui-finalproj-demo
Reduced version of final project that does not require specialized hardware. 
webcam_server.py is just to make the webcam available between windows and WSL, and most of the project logic (for this specific demo case) is located in mediapipe_hand_detector_gemma.py.

Important: This demo will likely only work on Windows concurrently with WSL, as that's the only platform setup I've tested on.


# Setup instructions
In both windows and WSL, create a python virtual environment in python 3.10.

In a windows terminal, run:
 ```
 pip install -r requirements_windows.txt
 ```

In a WSL Ubuntu terminal, run:
```
   pip install -r requirements_wsl.txt
```

In the windows terminal, start a webcam webook server:
 ```
 python webcam_server.py
 ```

Note the IP address that prints out. Put this IP address in mediapipe_hand_detector_gemma.py:
```pyhon
   SERVER_IP = '192.168.1.59'  # Change to match your server IP
   SERVER_PORT = 5000
```

In mediapipe_hand_detector_gemma.py, also set your huggingface API key (I will email one):
```
gemma_client = InferenceClient(
    provider="nebius",
    api_key="API_KEY_HERE",
)
```

In the WSL terminal, start the hand detector.

 ```
 python mediapipe_hand_detector_gemma.py
 ```
