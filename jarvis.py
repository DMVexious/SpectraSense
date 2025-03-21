import os
import sys
import queue
import json
import cv2
import time
import threading
import numpy as np
import pytesseract
import pygame
#from pyttsx3 import init as pyttsx3_init  
from picamera2 import Picamera2
from ultralytics import YOLO
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from imutils.object_detection import non_max_suppression
#from keras.models import load_model
#from keras.losses import MeanSquaredError
import subprocess
import re
import math
import tempfile
from gtts import gTTS

# (Optional) Force system packages if needed
os.environ["PYTHONPATH"] = "/usr/lib/python3/dist-packages"
sys.path.insert(0, "/usr/lib/python3/dist-packages")

AMP_THRESHOLD = 1000
MODEL_PATHS = {
    "speech": "/home/rvexi/blind/vosk-model-en-us-daanzu-20200905-lgraph",
    "object": "/home/rvexi/blind/yolo11s.pt",
    "currency": "/home/rvexi/blind/best (3).pt",
    "inversion": "/home/rvexi/blind/Invert_Model.h5",
    "east": "/home/rvexi/blind/frozen_east_text_detection.pb"
}

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "blocksize": 4000,
    "buffer_size": 3
}

VOICE_CONFIG = {
    "rate": 150,
    "volume": 0.8,
    "voice_id": "english_rp",
}


try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=2048)
except pygame.error as e:
    print(f"Primary audio init failed: {e}")
    try:
        pygame.mixer.init(frequency=22050, size=-8, channels=1, buffer=1024)
    except pygame.error as e:
        print(f"Audio initialization failed: {e}")
        sys.exit(1)

def send_to_ollama(command):
    import ollama
    MODEL = 'qwen2.5:0.5b'
    prompt = "Keep the response under 20 words: " + command
    res = ollama.generate(model=MODEL, prompt=prompt)
    return str(f"\n{res['response']}").strip()


# model_inversion = load_model(MODEL_PATHS["inversion"], custom_objects={'mse': MeanSquaredError()})

def check_and_correct_inversion(image, model):
    img_resized = cv2.resize(image, (28, 28))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)
    prediction = model.predict(img_input)
    is_inverted = prediction[0][0] > 0.5
    print(f"Inversion prediction: {prediction[0][0]} (Threshold: 0.5)")
    if is_inverted:
        print("Inverted image detected. Correcting inversion...")
        img_corrected = cv2.bitwise_not(img_gray)
    else:
        print("No inversion detected.")
        img_corrected = img_gray
    return img_corrected, is_inverted

def capture_image(save_path):
    try:
        subprocess.run(["rpicam-jpeg", "--output", save_path], check=True)
        return os.path.exists(save_path)
    except Exception as e:
        print("Error capturing image:", e)
        return False

def detect_text_block_with_rotated_rectangle(image, east_path):
    (H, W) = image.shape[:2]
    (newW, newH) = (1280, 1280)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    try:
        net = cv2.dnn.readNet(east_path)
    except Exception as e:
        print("Error loading EAST model:", e)
        return None
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                   (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    rects = []
    confidences = []
    (numRows, numCols) = scores.shape[2:4]
    for y in range(0, numRows):
        for x in range(0, numCols):
            if scores[0, 0, y, x] < 0.1:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            endX = int(offsetX + (cos * geometry[0, 1, y, x]) + (sin * geometry[0, 2, y, x]))
            endY = int(offsetY - (sin * geometry[0, 1, y, x]) + (cos * geometry[0, 2, y, x]))
            rects.append((endX - w, endY - h, endX, endY))
            confidences.append(scores[0, 0, y, x])
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    scaled_boxes = []
    for (startX, startY, endX, endY) in boxes:
        scaled_boxes.append((int(startX * rW), int(startY * rH),
                             int(endX * rW), int(endY * rH)))
    if scaled_boxes:
        all_points = []
        for (startX, startY, endX, endY) in scaled_boxes:
            all_points.extend([[startX, startY], [endX, startY], [endX, endY], [startX, endY]])
        all_points = np.array(all_points)
        rotated_rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rotated_rect)
        box = np.intp(box)
        return box
    return None

def calculate_skew_angle(box):
    box = sorted(box, key=lambda x: (x[1], x[0]))
    top_left, top_right, bottom_right, bottom_left = box
    dx = bottom_right[0] - bottom_left[0]
    dy = bottom_right[1] - bottom_left[1]
    if dx == 0:
        angle = 90
    else:
        angle = np.degrees(np.arctan(dy / dx))
    if angle < 0:
        angle += 180
    return angle if angle <= 90 else angle - 180

def correct_skew(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def draw_bounding_box(image, box, color=(0, 255, 0), thickness=2):
    cv2.drawContours(image, [box], -1, color, thickness)
    return image

def correct_skew_until_zero(image, east_path, max_iterations=5, angle_threshold=1.0, save_dir="/home/rvexi/Downloads/"):
    corrected_image = image.copy()
    final_box = None
    for i in range(max_iterations):
        box = detect_text_block_with_rotated_rectangle(corrected_image, east_path)
        if box is None:
            print(f"Iteration {i+1}: No text block detected. Stopping correction.")
            break
        draw_bounding_box(corrected_image, box)
        intermediate_path = os.path.join(save_dir, f"corrected_image_iteration_{i+1}.jpg")
        cv2.imwrite(intermediate_path, corrected_image)
        print(f"Saved intermediate corrected image at: {intermediate_path}")
        skew_angle = calculate_skew_angle(box)
        print(f"Iteration {i+1}: Calculated skew angle: {skew_angle:.2f} degrees")
        if abs(skew_angle) <= angle_threshold:
            print("Skew angle is within the acceptable range. Stopping corrections.")
            final_box = box
            return corrected_image, skew_angle, final_box
        corrected_image = correct_skew(corrected_image, skew_angle)
        rotated_path = os.path.join(save_dir, f"rotated_image_iteration_{i+1}.jpg")
        cv2.imwrite(rotated_path, corrected_image)
        print(f"Saved rotated image after correction at: {rotated_path}")
    print("Reached maximum iterations or acceptable skew angle threshold.")
    return corrected_image, skew_angle, final_box

def perform_text_recognition():
    try:
        east_path = MODEL_PATHS["east"]
        if not os.path.exists(east_path):
            print(f"Error: EAST model not found at {east_path}")
            return ""
        image_path = "captured_image.jpg"
        if not capture_image(image_path):
            print("Failed to capture image")
            return ""
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image")
            return ""
        corrected_image, final_angle, box = correct_skew_until_zero(image, east_path)
        cv2.imwrite("/home/rvexi/Downloads/corrected_image_final.jpg", corrected_image)
        print(f"Final skew angle: {final_angle:.2f} degrees")
        if box is not None:
            print("Text block detected. Extracting ROI for OCR.")
            pts = np.array(box, dtype="float32")
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            roi = corrected_image[y:y + h, x:x + w]
            roi_text = pytesseract.image_to_string(roi)
            print("\nOCR Text from ROI:")
            print(roi_text)
            return roi_text
        else:
            full_text = pytesseract.image_to_string(corrected_image)[:200]
            return full_text
    except Exception as e:
        print("Error during text recognition:", e)
        return ""

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATHS["object"])
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    
    def detect_objects(self, frame):
        try:
            results = self.model(frame, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                detections = []
                for i in range(len(results.boxes)):
                    cls_id = int(results.boxes.cls[i].item())
                    conf = results.boxes.conf[i].item()
                    detections.append(f"{results.names[cls_id]} ({conf:.2f})")
                print("Object detection debug:", ", ".join(detections))
                return "Detected: " + ", ".join(detections)
            else:
                print("Object detection debug: No objects detected")
                return ""
        except Exception as e:
            print("Error during object detection:", e)
            return ""

class CurrencyDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATHS["currency"])
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    
    def detect_currency(self, frame):
        try:
            results = self.model(frame, verbose=False)[0]
            if results.boxes is not None and len(results.boxes) > 0:
                cls_id = int(results.boxes.cls[0].item())
                conf = results.boxes.conf[0].item()
                print("Currency detection debug:", results.names[cls_id], f"({conf:.2f})")
                return f"Currency: {results.names[cls_id]} ({conf:.2f})"
            else:
                print("Currency detection debug: No currency detected")
                return ""
        except Exception as e:
            print("Error during currency detection:", e)
            return ""


def get_camera():
    try:
        cam = Picamera2()
        cam.preview_configuration.main.size = (640, 480)
        cam.preview_configuration.main.format = "RGB888"
        cam.preview_configuration.align()
        cam.configure("preview")
        return cam
    except Exception as e:
        print("Error acquiring camera:", e)
        return None


class VoiceAssistant:
    def __init__(self):
        self.wake_word = "jarvis"
        self.last_activation = 0
        self.activation_window = 5  # seconds between activations
        self.command_active = False
        self.listening_for_command = False
        self.muted = False
        self.audio_queue = queue.Queue(maxsize=AUDIO_CONFIG["buffer_size"])
        self.stop_event = threading.Event()
        self.continuous_thread = None

        self.model = Model(MODEL_PATHS["speech"])
        self.wake_recognizer = KaldiRecognizer(self.model, AUDIO_CONFIG["sample_rate"])
        self.wake_recognizer.SetWords(False)
        self.command_recognizer = KaldiRecognizer(self.model, AUDIO_CONFIG["sample_rate"])
        self.command_recognizer.SetWords(False)

        print("SYSTEM READY - Say the wake word ('jarvis') to activate")
        self.ready = True

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio error: {status}")
        self.audio_queue.put(indata.tobytes())

    def listen_loop(self):
        while not self.ready:
            time.sleep(0.1)
        print("Starting audio stream...")
        with sd.InputStream(
            callback=self.audio_callback,
            channels=AUDIO_CONFIG["channels"],
            samplerate=AUDIO_CONFIG["sample_rate"],
            blocksize=AUDIO_CONFIG["blocksize"],
            dtype="int16"
        ):
            while True:
                try:
                    data = self.audio_queue.get() 
                    frame = np.frombuffer(data, dtype=np.int16)
                    if np.max(np.abs(frame)) < AMP_THRESHOLD:
                        continue
                    print("\rListening...", end="")
                    if not self.listening_for_command:
                        if self.wake_recognizer.AcceptWaveform(data):
                            result = json.loads(self.wake_recognizer.Result())
                            text = result.get("text", "").lower()
                            if self.wake_word in text:
                                print("\nWAKE WORD DETECTED (final):", text)
                                self.interrupt_continuous_if_needed()
                                self.activate_command()
                        else:
                            partial = json.loads(self.wake_recognizer.PartialResult())
                            text = partial.get("partial", "").lower()
                            if self.wake_word in text:
                                print("\nWAKE WORD DETECTED (partial):", text)
                                self.interrupt_continuous_if_needed()
                                self.activate_command()
                except Exception as e:
                    print("Error in listen_loop:", e)

    def interrupt_continuous_if_needed(self):
        if self.continuous_thread is not None and self.continuous_thread.is_alive():
            print("Interrupting continuous detection...")
            self.stop_event.set()
            self.continuous_thread.join()
            self.stop_event.clear()

    def activate_command(self):
        if self.listening_for_command:
            return
        self.listening_for_command = True
        self.play_activation_sound()
        self.last_activation = time.time()
        print("WAKE WORD DETECTED")
        self.muted = True
        try:
            command_text = self.listen_for_command()
            self.execute_command(command_text)
        except Exception as e:
            print("Error in command activation:", e)
        self.wake_recognizer = KaldiRecognizer(self.model, AUDIO_CONFIG["sample_rate"])
        time.sleep(2)  # guard 
        self.muted = False
        self.listening_for_command = False

    def listen_for_command(self):
        duration = 5 
        self.speak("Listening for your command.")
        print("Recording command for 5 seconds...")
        audio_data = sd.rec(int(duration * AUDIO_CONFIG["sample_rate"]),
                            samplerate=AUDIO_CONFIG["sample_rate"],
                            channels=AUDIO_CONFIG["channels"],
                            dtype='int16')
        sd.wait()
        if self.command_recognizer.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(self.command_recognizer.Result())
        else:
            result = json.loads(self.command_recognizer.FinalResult())
        transcript = result.get("text", "")
        print("Transcribed command:", transcript)
        return transcript

    def play_activation_sound(self):
        try:
            pygame.mixer.music.load("/home/rvexi/blind/ui-wakesound-101soundboards.mp3")
            pygame.mixer.music.play()
            time.sleep(0.3)
        except Exception as e:
            print("Error playing activation sound:", e)

    def gtts_speak(self, text):

        try:
            tts = gTTS(text=text, lang='en', tld='com.au') 
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_filename = fp.name
                tts.write_to_fp(fp)
            os.system(f"mpg123 {temp_filename} > /dev/null 2>&1")
            os.remove(temp_filename)
        except Exception as e:
            print("gTTS error:", e)

    def speak(self, text):
        self.gtts_speak(text)

    def execute_command(self, command):
        self.show_thinking_effect()
        command = command.strip().lower()
        try:
            if command.startswith("perform"):
                stripped = command[len("perform"):].strip()
                if any(kw in stripped for kw in ["detect", "object"]):
                    self.handle_object_command_continuous()
                    return
                elif any(kw in stripped for kw in ["currency", "money"]):
                    self.handle_currency_command_continuous()
                    return
                elif any(kw in stripped for kw in ["read", "text"]):
                    self.handle_text_command()
                    return
            if any(kw in command for kw in ["text", "read"]):
                self.handle_text_command()
            elif any(kw in command for kw in ["currency", "money"]):
                self.handle_currency_command_continuous()
            elif any(kw in command for kw in ["object", "detect"]):
                self.handle_object_command_continuous()
            else:
                self.handle_llm_command(command)
        except Exception as e:
            print("Error executing command:", e)

    def handle_text_command(self):
        try:
            self.speak("Scanning text...")
            ocr_text = perform_text_recognition()
            if ocr_text:
                self.speak(f"Found text: {ocr_text[:100]}")
            else:
                self.speak("No text detected.")
        except Exception as e:
            print("Error in text command:", e)

    def handle_object_command_continuous(self):
        self.speak("Starting continuous object detection. Say 'jarvis' to stop.")
        def continuous_object_detection():
            cam = get_camera()
            if cam is None:
                self.speak("Failed to acquire camera for object detection.")
                return
            try:
                cam.start()
                detector = ObjectDetector()
                last_announced = ""
                while not self.stop_event.is_set():
                    try:
                        frame = cam.capture_array()
                    except Exception as e:
                        print("Error capturing frame:", e)
                        cam = get_camera()
                        if cam is None:
                            break
                        cam.start()
                        continue
                    result_text = detector.detect_objects(frame)
                    if result_text and result_text != last_announced:
                        print("Object detection debug:", result_text)
                        self.speak(result_text)
                        last_announced = result_text
                    time.sleep(0.5)
            except Exception as e:
                print("Error in continuous object detection:", e)
            finally:
                try:
                    cam.stop()
                    cam.close()
                except Exception as e:
                    print("Error releasing camera in object detection:", e)
                self.speak("Stopping continuous object detection.")
        self.stop_event.clear()
        self.continuous_thread = threading.Thread(target=continuous_object_detection)
        self.continuous_thread.start()

    def handle_currency_command_continuous(self):
        self.speak("Starting continuous currency detection. Say 'jarvis' to stop.")
        def continuous_currency_detection():
            cam = get_camera()
            if cam is None:
                self.speak("Failed to acquire camera for currency detection.")
                return
            try:
                cam.start()
                detector = CurrencyDetector()
                last_announced = ""
                while not self.stop_event.is_set():
                    try:
                        frame = cam.capture_array()
                    except Exception as e:
                        print("Error capturing frame:", e)
                        cam = get_camera()
                        if cam is None:
                            break
                        cam.start()
                        continue
                    result_text = detector.detect_currency(frame)
                    if result_text and result_text != last_announced:
                        print("Currency detection debug:", result_text)
                        self.speak(result_text)
                        last_announced = result_text
                    time.sleep(0.5)
            except Exception as e:
                print("Error in continuous currency detection:", e)
            finally:
                try:
                    cam.stop()
                    cam.close()
                except Exception as e:
                    print("Error releasing camera in currency detection:", e)
                self.speak("Stopping continuous currency detection.")
        self.stop_event.clear()
        self.continuous_thread = threading.Thread(target=continuous_currency_detection)
        self.continuous_thread.start()

    def handle_llm_command(self, command):
        try:
            response = send_to_ollama(command)
            self.speak(response)
        except Exception as e:
            print("Error in LLM command:", e)

    def show_thinking_effect(self):
        print("[\u25B6]", end=" ", flush=True)

def get_camera():
    try:
        cam = Picamera2()
        cam.preview_configuration.main.size = (640, 480)
        cam.preview_configuration.main.format = "RGB888"
        cam.preview_configuration.align()
        cam.configure("preview")
        return cam
    except Exception as e:
        print("Error acquiring camera:", e)
        return None


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.listen_loop()
