import cv2
import numpy as np
import requests
import time
import threading
import queue
import uuid
import os
import json
from io import BytesIO
from PIL import Image
import mediapipe as mp
from collections import deque
import math
import base64
from huggingface_hub import InferenceClient

# server settings
SERVER_IP = '192.168.1.59' #THIS IS FOR WEBCAM
SERVER_PORT = 5000

# serial server settings
SERIAL_SERVER_IP = '192.168.1.59' #THIS IS FOR VR GLOVE
SERIAL_SERVER_PORT = 5001

# init gemma client
gemma_client = InferenceClient(
    provider="nebius",
    api_key="API_KEY_HERE", #FILL THIS IN
)

# init mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def send_serial_command(command):
    """send command to serial server via http"""
    try:
        response = requests.post(
            f'http://{SERIAL_SERVER_IP}:{SERIAL_SERVER_PORT}/send',
            json={"command": command},
            timeout=1.0
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('success', False)
        else:
            print(f"Error: Serial server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to serial server: {e}")
        return False

def check_serial_status():
    """check if serial server is available"""
    try:
        response = requests.get(
            f'http://{SERIAL_SERVER_IP}:{SERIAL_SERVER_PORT}/status',
            timeout=1.0
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('enabled', False)
        else:
            return False
    except Exception as e:
        print(f"Error checking serial server status: {e}")
        return False

# gemma message template
gemma_message_template = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Respond Y or N for if 1st person is holding anything. Only say yes if you can actually see the handheld object. N for empty hands."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": ""
            }
        }
    ]
}

# global vars for gemma results
gemma_result_lock = threading.Lock()
gemma_response = "N/A"
gemma_latency = 0.0
gemma_last_update = 0

# track gemma responses and status
gemma_responses = {}
gemma_latencies = {}
gemma_timestamps = {}
active_requests = set()
recent_latencies = deque(maxlen=10)

# queue for frames to process
gemma_queue = queue.Queue(maxsize=10)

# serial flag
serial_available = False

def encode_image(image_array):
    """convert image to base64"""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')

def get_average_latency():
    """get avg latency"""
    if not recent_latencies:
        return 2.0
    return sum(recent_latencies) / len(recent_latencies)

def gemma_worker():
    """background thread for gemma api processing"""
    global gemma_response, gemma_latency, gemma_last_update
    global gemma_responses, gemma_latencies, gemma_timestamps, active_requests
    
    print("Gemma worker thread started")
    
    while True:
        try:
            with gemma_result_lock:
                current_active = len(active_requests)
            
            if current_active < 3:
                try:
                    item = gemma_queue.get(timeout=0.1)
                    
                    if item is None:
                        break
                        
                    frame, request_id = item
                    
                    threading.Thread(
                        target=process_gemma_request, 
                        args=(frame, request_id), 
                        daemon=True
                    ).start()
                    
                except queue.Empty:
                    pass
            else:
                time.sleep(0.1)
                
            # clean up old responses
            with gemma_result_lock:
                current_time = time.time()
                old_requests = [req_id for req_id, ts in gemma_timestamps.items() 
                                if current_time - ts > 10.0 and req_id not in active_requests]
                
                for req_id in old_requests:
                    if req_id in gemma_responses:
                        del gemma_responses[req_id]
                    if req_id in gemma_latencies:
                        del gemma_latencies[req_id]
                    if req_id in gemma_timestamps:
                        del gemma_timestamps[req_id]
            
        except Exception as e:
            print(f"Error in Gemma worker main loop: {e}")
            time.sleep(0.5)
            
    print("Gemma worker thread stopped")

def process_gemma_request(frame, request_id):
    """process single gemma request"""
    global gemma_responses, gemma_latencies, gemma_timestamps, active_requests, recent_latencies
    
    try:
        with gemma_result_lock:
            active_requests.add(request_id)
        
        base64_image = encode_image(frame)
        
        message = dict(gemma_message_template)
        message["content"] = list(gemma_message_template["content"])
        message["content"][1] = dict(gemma_message_template["content"][1])
        message["content"][1]["image_url"] = {"url": f"data:image/png;base64,{base64_image}"}
        
        start_time = time.time()
        
        try:
            completion = gemma_client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=[message],
                max_tokens=512,
            )
            
            response_text = completion.choices[0].message.content
            
            end_time = time.time()
            latency = end_time - start_time
            
            with gemma_result_lock:
                gemma_responses[request_id] = response_text.strip()
                gemma_latencies[request_id] = latency
                gemma_timestamps[request_id] = end_time
                
                gemma_response = response_text.strip()
                gemma_latency = latency
                gemma_last_update = end_time
                
                recent_latencies.append(latency)
            
            print(f"Gemma response {request_id}: {response_text} (latency: {latency:.2f}s)")
            
        except Exception as e:
            print(f"Error with Gemma API call {request_id}: {e}")
            
    except Exception as e:
        print(f"Error processing Gemma request {request_id}: {e}")
    finally:
        gemma_queue.task_done()
        with gemma_result_lock:
            if request_id in active_requests:
                active_requests.remove(request_id)

def get_frame_from_server():
    """get frame from webcam server"""
    try:
        response = requests.get(f'http://{SERVER_IP}:{SERVER_PORT}/single_frame')
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return frame
        else:
            print(f"Error: Server returned status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return None

def calculate_hand_features(hand_landmarks, image_shape):
    """calc hand features for detection"""
    h, w = image_shape[:2]
    
    landmarks_px = []
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        landmarks_px.append((x, y))
    
    wrist = landmarks_px[mp_hands.HandLandmark.WRIST]
    thumb_tip = landmarks_px[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks_px[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks_px[mp_hands.HandLandmark.PINKY_TIP]
    
    palm_landmarks = [
        landmarks_px[mp_hands.HandLandmark.THUMB_MCP],
        landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.RING_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.PINKY_MCP]
    ]
    palm_center = np.mean(palm_landmarks, axis=0).astype(int)
    
    dx = palm_center[0] - wrist[0]
    dy = palm_center[1] - wrist[1]
    hand_angle = math.degrees(math.atan2(dy, dx))
    
    thumb_joints = [
        landmarks_px[mp_hands.HandLandmark.THUMB_CMC],
        landmarks_px[mp_hands.HandLandmark.THUMB_MCP],
        landmarks_px[mp_hands.HandLandmark.THUMB_IP],
        landmarks_px[mp_hands.HandLandmark.THUMB_TIP]
    ]
    
    index_joints = [
        landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_PIP],
        landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_DIP],
        landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ]
    
    middle_joints = [
        landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
        landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
        landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ]
    
    ring_joints = [
        landmarks_px[mp_hands.HandLandmark.RING_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.RING_FINGER_PIP],
        landmarks_px[mp_hands.HandLandmark.RING_FINGER_DIP],
        landmarks_px[mp_hands.HandLandmark.RING_FINGER_TIP]
    ]
    
    pinky_joints = [
        landmarks_px[mp_hands.HandLandmark.PINKY_MCP],
        landmarks_px[mp_hands.HandLandmark.PINKY_PIP],
        landmarks_px[mp_hands.HandLandmark.PINKY_DIP],
        landmarks_px[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    finger_curls = []
    
    thumb_curl = calculate_thumb_curl(thumb_joints, wrist, palm_center)
    finger_curls.append(thumb_curl)
    
    finger_curls.append(calculate_finger_curl(index_joints, palm_center))
    finger_curls.append(calculate_finger_curl(middle_joints, palm_center))
    finger_curls.append(calculate_finger_curl(ring_joints, palm_center))
    finger_curls.append(calculate_finger_curl(pinky_joints, palm_center))
    
    fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    finger_mcps = [
        landmarks_px[mp_hands.HandLandmark.THUMB_MCP],
        landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.RING_FINGER_MCP],
        landmarks_px[mp_hands.HandLandmark.PINKY_MCP]
    ]
    
    tip_to_palm_distances = []
    for tip in fingertips:
        dist = np.linalg.norm(np.array(tip) - np.array(palm_center))
        tip_to_palm_distances.append(dist)
    
    mcp_to_palm_distances = []
    for mcp in finger_mcps:
        dist = np.linalg.norm(np.array(mcp) - np.array(palm_center))
        mcp_to_palm_distances.append(dist)
    avg_mcp_to_palm = np.mean(mcp_to_palm_distances)
    
    curled_fingers = []
    for i, (tip, mcp) in enumerate(zip(fingertips, finger_mcps)):
        if tip_to_palm_distances[i] < mcp_to_palm_distances[i] * 1.2:
            curled_fingers.append(i)
    
    thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
    thumb_middle_dist = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))
    pinch_threshold = avg_mcp_to_palm * 0.25
    is_pinching = thumb_index_dist < pinch_threshold or thumb_middle_dist < pinch_threshold
    
    thumb_pos = np.array(thumb_tip)
    other_finger_pos = np.mean([index_tip, middle_tip], axis=0)
    grip_center = ((thumb_pos + other_finger_pos) / 2).astype(int)
    
    grip_size = np.linalg.norm(thumb_pos - other_finger_pos)
    
    if is_pinching or len(curled_fingers) >= 2:
        holding_confidence = min(1.0, (len(curled_fingers) / 5) + (1 if is_pinching else 0) * 0.5)
    else:
        holding_confidence = 0
    
    return {
        "palm_center": tuple(palm_center),
        "grip_center": tuple(grip_center),
        "grip_size": grip_size,
        "hand_angle": hand_angle,
        "curled_fingers": curled_fingers,
        "is_pinching": is_pinching,
        "holding_confidence": holding_confidence,
        "landmarks_px": landmarks_px,
        "finger_curls": finger_curls
    }

def calculate_thumb_curl(thumb_joints, wrist, palm_center):
    """calc thumb curl (0=curled, 1=extended)"""
    angle1 = calculate_angle(thumb_joints[0], thumb_joints[1], thumb_joints[2])
    angle2 = calculate_angle(thumb_joints[1], thumb_joints[2], thumb_joints[3])
    
    tip_to_palm_dist = np.linalg.norm(np.array(thumb_joints[3]) - np.array(palm_center))
    mcp_to_palm_dist = np.linalg.norm(np.array(thumb_joints[1]) - np.array(palm_center))
    
    dist_ratio = min(tip_to_palm_dist / (mcp_to_palm_dist * 1.5), 1.0)
    
    curl_from_angles = 1.0 - (min(angle1, 90) / 90.0 * 0.5 + min(angle2, 90) / 90.0 * 0.5)
    
    thumb_curl = dist_ratio * 0.7 + (1.0 - curl_from_angles) * 0.3
    
    return min(max(thumb_curl, 0.0), 1.0)

def calculate_finger_curl(finger_joints, palm_center):
    """calc finger curl (0=curled, 1=extended)"""
    angle1 = calculate_angle(finger_joints[0], finger_joints[1], finger_joints[2])
    angle2 = calculate_angle(finger_joints[1], finger_joints[2], finger_joints[3])
    
    tip_to_palm_dist = np.linalg.norm(np.array(finger_joints[3]) - np.array(palm_center))
    mcp_to_palm_dist = np.linalg.norm(np.array(finger_joints[0]) - np.array(palm_center))
    
    dist_ratio = min(tip_to_palm_dist / (mcp_to_palm_dist * 2.0), 1.0)
    
    angle_extension = 1.0 - ((angle1 + angle2) / 180.0)
    
    finger_curl = dist_ratio * 0.7 + angle_extension * 0.3
    
    return min(max(finger_curl, 0.0), 1.0)

def calculate_angle(point1, point2, point3):
    """calc angle between 3 points in degrees"""
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def create_grip_mask(hand_features, frame_shape):
    """create mask for held object area"""
    h, w = frame_shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    grip_center = hand_features["grip_center"]
    grip_size = hand_features["grip_size"]
    palm_center = hand_features["palm_center"]
    landmarks_px = hand_features["landmarks_px"]
    
    thumb_tip = landmarks_px[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks_px[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks_px[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    grip_polygon = np.array([thumb_tip, index_tip, middle_tip, palm_center], dtype=np.int32)
    cv2.fillConvexPoly(mask, grip_polygon, 255)
    
    radius = int(grip_size / 2)
    cv2.circle(mask, grip_center, radius, 255, -1)
    
    return mask

def advanced_object_detection(frame, hand_landmarks, gemma_says_holding=False):
    """detect objects in hand with multiple techniques"""
    if frame is None or hand_landmarks is None:
        return None, None, None, None, 0.0
    
    h, w = frame.shape[:2]
    
    debug_vis = frame.copy()
    
    hand_features = calculate_hand_features(hand_landmarks, frame.shape)
    
    grip_mask = create_grip_mask(hand_features, frame.shape)
    
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    hand_hull = cv2.convexHull(np.array(hand_features["landmarks_px"]))
    cv2.fillConvexPoly(hand_mask, hand_hull, 255)
    
    palm_center = hand_features["palm_center"]
    grip_center = hand_features["grip_center"]
    cv2.circle(debug_vis, palm_center, 5, (255, 0, 0), -1)
    cv2.circle(debug_vis, grip_center, 5, (0, 255, 255), -1)
    
    holding_confidence = hand_features["holding_confidence"]
    cv2.putText(debug_vis, f"hold conf: {holding_confidence:.2f}", (10, 290), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if gemma_says_holding:
        cv2.putText(debug_vis, "gemma override active", (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    threshold = 0.1 if gemma_says_holding else 0.2
    if holding_confidence < threshold:
        return None, grip_mask, debug_vis, None, holding_confidence
    
    adjusted_confidence = holding_confidence
    if gemma_says_holding and holding_confidence < 0.5:
        adjusted_confidence = min(holding_confidence + 0.15, 0.6)
        cv2.putText(debug_vis, f"adjusted conf: {adjusted_confidence:.2f}", (10, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    grabcut_mask = np.zeros((h, w), dtype=np.uint8)
    grabcut_mask.fill(cv2.GC_BGD)
    
    grabcut_mask[grip_mask > 0] = cv2.GC_PR_FGD
    
    for point in hand_features["landmarks_px"]:
        if grip_mask[point[1], point[0]] == 0:
            cv2.circle(grabcut_mask, point, 3, cv2.GC_BGD, -1)
    
    grip_center = hand_features["grip_center"]
    grip_size = hand_features["grip_size"]
    inner_radius = int(grip_size / 4)
    cv2.circle(grabcut_mask, grip_center, inner_radius, cv2.GC_FGD, -1)
    
    grabcut_vis = np.zeros((h, w, 3), dtype=np.uint8)
    grabcut_vis[grabcut_mask == cv2.GC_BGD] = [0, 0, 100]
    grabcut_vis[grabcut_mask == cv2.GC_PR_BGD] = [0, 0, 255]
    grabcut_vis[grabcut_mask == cv2.GC_PR_FGD] = [0, 255, 0]
    grabcut_vis[grabcut_mask == cv2.GC_FGD] = [0, 255, 255]
    
    debug_vis = cv2.addWeighted(debug_vis, 0.7, grabcut_vis, 0.3, 0)
    
    x, y, w_rect, h_rect = cv2.boundingRect(np.array(np.where(grip_mask > 0)[::-1]).T)
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w_rect = min(w - x, w_rect + 2*padding)
    h_rect = min(h - y, h_rect + 2*padding)
    rect = (x, y, w_rect, h_rect)
    
    cv2.rectangle(debug_vis, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 2)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    min_grip_area = 100
    if np.count_nonzero(grip_mask) < min_grip_area:
        return None, grip_mask, debug_vis, None, adjusted_confidence
    
    try:
        cv2.grabCut(frame, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f"grabcut error: {e}")
        return None, grip_mask, debug_vis, None, adjusted_confidence
    
    foreground_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 50
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
            
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
        
        if grip_mask[cy, cx] > 0:
            dist_to_grip = np.linalg.norm(np.array(centroid) - np.array(grip_center))
            
            if dist_to_grip < grip_size:
                valid_contours.append((contour, area, dist_to_grip))
    
    valid_contours.sort(key=lambda x: x[2])
    
    object_contour = None
    if valid_contours:
        object_contour = valid_contours[0][0]
        
        cv2.drawContours(debug_vis, [object_contour], 0, (0, 255, 255), 2)
        
        contour_area = cv2.contourArea(object_contour)
        expected_area = np.pi * (grip_size/2)**2
        
        area_ratio = min(contour_area, expected_area) / max(contour_area, expected_area)
        
        segmented_obj = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        
        foreground_pixels = frame[foreground_mask > 0]
        
        color_variance = 0
        if len(foreground_pixels) > 0:
            color_variance = np.var(foreground_pixels, axis=0).mean()
        
        base_confidence = (adjusted_confidence * 0.6 + 
                          area_ratio * 0.2 + 
                          min(1.0, color_variance / 1000) * 0.2)
                          
        object_confidence = min(base_confidence * 1.2, 1.0) if gemma_says_holding else base_confidence
        
        cv2.putText(debug_vis, f"obj conf: {object_confidence:.2f}", (10, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        detection_threshold = 0.3 if gemma_says_holding else 0.4
        if object_confidence < detection_threshold:
            return None, grip_mask, debug_vis, None, adjusted_confidence
            
        return object_contour, foreground_mask, debug_vis, segmented_obj, object_confidence
    
    return None, grip_mask, debug_vis, None, adjusted_confidence

def process_frame(frame, hands, previous_command=None):
    """process frame with mediapipe hands"""
    global gemma_response, gemma_latency, gemma_last_update
    
    if frame is None:
        return None, None, False, False, None, None, 0.0, None
    
    gemma_says_holding = False
    
    with gemma_result_lock:
        current_time = time.time()
        
        recent_responses = []
        for req_id, timestamp in gemma_timestamps.items():
            age = current_time - timestamp
            if age < 5.0:
                response = gemma_responses.get(req_id, "")
                recent_responses.append((req_id, response, age))
        
        recent_responses.sort(key=lambda x: x[2])
        
        if recent_responses:
            most_recent = recent_responses[0]
            gemma_says_holding = most_recent[1].lower().startswith('y')
            
            gemma_response = most_recent[1]
            gemma_latency = gemma_latencies.get(most_recent[0], 0.0)
            gemma_last_update = gemma_timestamps.get(most_recent[0], 0.0)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    
    annotated_frame = frame.copy()
    
    object_contour = None
    holding_something = False
    object_detected = False
    mask_vis = None
    segmented_obj = None
    object_confidence = 0.0
    finger_curls = [0, 0, 0, 0, 0]
    
    hand_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
    
    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            obj_contour, obj_mask, debug_vis, obj_segment, conf = advanced_object_detection(
                frame, hand_landmarks, gemma_says_holding)
            mask_vis = debug_vis
            segmented_obj = obj_segment
            object_confidence = conf
            
            hand_features = calculate_hand_features(hand_landmarks, frame.shape)
            finger_curls = hand_features["finger_curls"]
            
            holding_something = conf > 0.3
            
            if obj_contour is not None:
                cv2.drawContours(annotated_frame, [obj_contour], 0, (0, 255, 255), 2)
                object_contour = obj_contour
                object_detected = True
    
    if hand_detected and (holding_something or gemma_says_holding):
        haptic_values = []
        for curl in finger_curls:
            scaled_value = int((1.0 - curl) * 1000)
            scaled_value = max(0, min(1000, scaled_value))
            haptic_values.append(scaled_value)
            
        serial_command = f"A{haptic_values[0]}B{haptic_values[1]}C{haptic_values[2]}D{haptic_values[3]}E{haptic_values[4]}"
    else:
        serial_command = "A0B0C0D0E0"
    
    command_changed = serial_command != previous_command
    
    if serial_available and command_changed:
        send_serial_command(serial_command)
    
    hand_count = 0 if not hand_detected else len(results.multi_hand_landmarks)
    cv2.putText(annotated_frame, f"hands: {hand_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(annotated_frame, f"holding: {'yes' if holding_something else 'no'}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    status_text = f"mediapipe: {'yes' if object_detected else 'no'}"
    if object_detected:
        status_text += f" ({object_confidence:.2f})"
    cv2.putText(annotated_frame, status_text, (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    with gemma_result_lock:
        gemma_text = f"gemma: {gemma_response}"
        if gemma_latency > 0:
            gemma_text += f" ({gemma_latency:.2f}s)"
        time_since_update = time.time() - gemma_last_update
        
        active_count = len(active_requests)
    
    cv2.putText(annotated_frame, gemma_text, (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if time_since_update > 5 and gemma_response != "N/A":
        cv2.putText(annotated_frame, "[old result]", (400, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(annotated_frame, f"active gemma: {active_count}/3", (10, 390), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for i, curl in enumerate(finger_curls):
        finger_name = ["thumb", "index", "middle", "ring", "pinky"][i]
        cv2.putText(annotated_frame, f"{finger_name}: {curl:.2f}", (500, 50 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    serial_status = "connected" if serial_available else "disconnected"
    cv2.putText(annotated_frame, f"serial: {serial_status}", (10, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cmd_status = f"cmd: {serial_command}"
    if command_changed:
        cmd_status += " [sent]"
    else:
        cmd_status += " [unchanged]"
    cv2.putText(annotated_frame, cmd_status, (10, 470), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    agreement = False
    if (gemma_response.lower().startswith('y') and object_detected) or \
       (gemma_response.lower().startswith('n') and not object_detected):
        agreement = True
    
    if not agreement and gemma_response != "N/A":
        cv2.putText(annotated_frame, "disagreement", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return annotated_frame, results, holding_something, object_detected, mask_vis, segmented_obj, object_confidence, serial_command

def main():
    """main application loop"""
    global serial_available
    
    print("mediapipe hand detector with gemma starting...")
    print(f"connecting to webcam server at {SERVER_IP}:{SERVER_PORT}")
    print(f"checking serial server at {SERIAL_SERVER_IP}:{SERIAL_SERVER_PORT}")
    print("press 'q' to quit")
    
    serial_available = check_serial_status()
    if serial_available:
        print("successfully connected to serial server")
    else:
        print("could not connect to serial server")
    
    cv2.namedWindow("MediaPipe Hands")
    cv2.namedWindow("Detection Process")
    cv2.namedWindow("Segmented Object")
    
    gemma_thread = threading.Thread(target=gemma_worker, daemon=True)
    gemma_thread.start()
    
    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.3,
        max_num_hands=2) as hands:
        
        confidence_history = deque(maxlen=10)
        last_holding_states = deque(maxlen=10)
        last_object_states = deque(maxlen=10)
        
        last_gemma_frame_time = 0
        frame_count = 0
        
        last_serial_check = 0
        
        previous_serial_command = None
        
        last_state_change_time = 0
        debounce_time = 0.5
        
        while True:
            start_time = time.time()
            frame_count += 1
            
            if time.time() - last_serial_check > 5:
                serial_available = check_serial_status()
                last_serial_check = time.time()
            
            frame = get_frame_from_server()
            if frame is None:
                print("failed to get frame, retrying...")
                time.sleep(1)
                continue
            
            annotated_frame, results, holding_something, object_detected, mask_vis, segmented_obj, confidence, current_serial_command = process_frame(frame, hands, previous_serial_command)
            
            command_changed = current_serial_command != previous_serial_command
            current_time = time.time()
            
            if command_changed and (current_time - last_state_change_time) > debounce_time:
                previous_serial_command = current_serial_command
                last_state_change_time = current_time
                
                print(f"command changed to: {current_serial_command}")
            
            avg_latency = get_average_latency()
            target_spacing = max(0.3, avg_latency / 3.0)
            
            with gemma_result_lock:
                active_count = len(active_requests)
            
            if (results.multi_hand_landmarks and 
                active_count < 3 and 
                current_time - last_gemma_frame_time > target_spacing):
                
                try:
                    request_id = str(uuid.uuid4())
                    
                    gemma_queue.put_nowait((frame.copy(), request_id))
                    
                    last_gemma_frame_time = current_time
                    
                    print(f"frame {frame_count} sent to gemma (spacing: {target_spacing:.2f}s)")
                except queue.Full:
                    pass
            
            confidence_history.append(confidence)
            last_holding_states.append(holding_something)
            last_object_states.append(object_detected)
            
            avg_confidence = sum(confidence_history) / len(confidence_history) if confidence_history else 0
            smoothed_holding = sum(last_holding_states) / len(last_holding_states) > 0.5 if last_holding_states else False
            smoothed_object = sum(last_object_states) / len(last_object_states) > 0.5 if last_object_states else False
            
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            cv2.putText(annotated_frame, f"fps: {fps:.1f}", (10, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            smoothed_status = f"smoothed: {'yes' if smoothed_object else 'no'}"
            if smoothed_object:
                smoothed_status += f" ({avg_confidence:.2f})"
            cv2.putText(annotated_frame, smoothed_status, (10, 270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"avg latency: {avg_latency:.2f}s", (10, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"target spacing: {target_spacing:.2f}s", (10, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("MediaPipe Hands", annotated_frame)
            if mask_vis is not None:
                cv2.imshow("Detection Process", mask_vis)
            if segmented_obj is not None:
                cv2.imshow("Segmented Object", segmented_obj)
            else:
                blank = np.zeros_like(frame)
                cv2.putText(blank, "no object detected", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Segmented Object", blank)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    try:
        gemma_queue.put(None)
        gemma_thread.join(timeout=1.0)
    except:
        pass
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 