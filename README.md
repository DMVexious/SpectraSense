# Spectra Sense

**Spectra Sense** is an IoT device designed to assist the visually impaired by recognizing visual patterns and conveying information through audio feedback.
It leverages a **Raspberry Pi 5**, a **Raspberry Pi Camera Module 3**, and a set of earpods (or any compatible headphones) to provide real-time assistance. 
This project was developed for the **YISF** competition, where it received a **Silver** placement.

## Table of Contents
1. [Features](#features)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Requirements](#software-requirements)
4. [Installation & Setup](#installation--setup)
5. [Project Structure](#project-structure)
6. [Usage](#usage)
7. [Commands Overview](#commands-overview)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [Acknowledgments](#acknowledgments)

---

## Features

1. **Text Recognition (OCR):**  
   Uses PyTesseract and the EAST text detection model to recognize text from captured images, allowing users to read books or documents aloud.

2. **Currency Recognition:**  
   Identifies Indonesian currency bills using a custom-trained YOLO model.

3. **Object Recognition:**  
   Utilizes a YOLO model to detect various objects in the camera’s field of view.

4. **Voice Assistant Integration:**  
   - **Vosk** for offline speech recognition (wake-word detection, commands).  
   - **gTTS** (Google Text-to-Speech) for generating audio responses.  
   - **ollama** with the Qwen (0.5b) model for question answering.

5. **Inversion Correction & Skew Detection:**  
   - Detects whether images or text are inverted and corrects them before performing OCR.  
   - Adjusts the skew angle of text regions using the EAST model to improve recognition accuracy.

6. **Real-time Feedback:**  
   Provides immediate audio feedback when detecting text, currency, or objects.

---

## Hardware Requirements

- **Raspberry Pi 5** (or similar board with enough processing power).
- **Raspberry Pi Camera Module 3** (or another compatible camera).
- **Earbuds/Headphones** for audio feedback.
- Optional: External microphone if not using Pi’s built-in audio input (depends on your setup).

---

## Software Requirements

- **Python 3.7+**
- **pip** (Python package manager)
- **Git** (to clone the repository or manage code versions)
- **Git LFS** (for large model files, if you are storing them in this repo)

### Python Libraries
Below are some key Python libraries used in `jarvis.py`:

1. **OpenCV (cv2)**
2. **Numpy**
3. **Pytesseract**  
4. **PyGame** (for audio playback)
5. **Vosk** (speech recognition)
6. **SoundDevice** (audio capture)
7. **gTTS** (text-to-speech)
8. **ollama** (for Qwen LLM queries)
9. **Ultralytics (YOLO)**
10. **(Optional) TensorFlow/Keras** for advanced image processing or model usage

You can install dependencies with:
```bash
pip install -r requirements.txt
```
(If you have a `requirements.txt` file in your project; otherwise, install the above packages individually.)

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DMVexious/SpectraSense.git
   cd SpectraSense
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   or install the individual libraries manually.

3. **Set Up Model Files**
   - **YOLO model** (`yolo11s.pt`)  
   - **Currency model** (`currency.pt`)  
   - **EAST text detection model** (`frozen_east_text_detection.pb`)  
   - **Inversion model** (`Invert_Model.h5`)  

   Make sure these files are in the same directory as `jarvis.py` or update the `MODEL_PATHS` in `jarvis.py` accordingly.

4. **Configure Audio**
   - Ensure `pygame.mixer` and `sounddevice` can access your audio hardware.  
   - If using a USB mic or Pi’s built-in audio input, verify it works with `sounddevice`.

5. **(Optional) Tesseract Installation**  
   - If PyTesseract can’t find the Tesseract engine, install Tesseract OCR. On Raspberry Pi OS (Debian-based):
     ```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr
     ```

6. **(Optional) Ollama Installation**  
   - If you want to use the Qwen (0.5b) model for advanced queries, install and configure [ollama](https://github.com/jmorganca/ollama) or use the Python `ollama` library.  

---

## Project Structure

```
SpectraSense/
│
├── jarvis.py                    # Main Python script containing the voice assistant logic
├── yolo11s.pt                   # YOLO object detection model
├── currency.pt                  # YOLO model for Indonesian currency detection
├── Invert_Model.h5              # Model to detect and correct image inversion
├── frozen_east_text_detection.pb # EAST text detection model for OCR
├── ui-wakesound-101soundboards.mp3 # Audio file for wake word confirmation
├── requirements.txt             # List of Python dependencies (if present)
├── README.md                    # Project documentation
└── ...                          # Other supporting files
```

---

## Usage

1. **Connect Your Hardware**  
   - Attach the Raspberry Pi Camera Module 3.  
   - Plug in headphones/earpods or speakers.

2. **Run the Main Script**  
   From your terminal in the project directory:
   ```bash
   python jarvis.py
   ```
   - The script initializes the Vosk model, sets up the camera, and waits for the wake word (“jarvis”).

3. **Use Voice Commands**  
   - Say “jarvis” to wake up the assistant.  
   - After the wake sound, speak your command (e.g., “Perform text recognition,” “Perform object detection,” “Perform currency recognition,” etc.).  

4. **Stop Continuous Detection**  
   - To stop continuous object or currency detection, say “jarvis” again. The assistant will interrupt the current loop.

---

## Commands Overview

- **“jarvis”**: Wake word to activate the assistant.
- **“Perform text recognition”** or **“Read text”**:  
  - Captures an image, corrects skew/inversion, and performs OCR.  
  - Reads aloud any detected text.
- **“Perform object detection”** or **“Detect object”**:  
  - Continuously detects objects in the camera feed using YOLO.  
  - Speaks out identified objects.
- **“Perform currency recognition”** or **“Money”**:  
  - Continuously detects Indonesian bills using the currency YOLO model.
- **Any other query**:  
  - Sends the query to the ollama Qwen (0.5b) model for a short text response.

---

## Troubleshooting

1. **No Audio Output**  
   - Check `pygame.mixer` initialization in `jarvis.py`.  
   - Verify your headphones or speakers are properly connected.

2. **Camera Issues**  
   - Ensure the camera is enabled in `raspi-config`.  
   - Check that `picamera2` is installed and recognized.

3. **Speech Recognition Not Working**  
   - Make sure your microphone is working and accessible to `sounddevice`.  
   - Check that the Vosk model path in `MODEL_PATHS["speech"]` is correct.

4. **OCR Inaccurate**  
   - Verify Tesseract is installed.  
   - Ensure `pytesseract.pytesseract.tesseract_cmd` points to the correct Tesseract binary if needed.


---

## Contributing

Contributions are welcome! Feel free to open issues for feature requests, bug reports, or general improvements. Fork the repo and submit pull requests for any enhancements you’d like to see.

## Acknowledgments

- **YISF Competition** for the opportunity to present this project (Silver placement).
- **Vosk** for offline speech recognition.
- **OpenCV**, **PyTesseract**, and **EAST** for OCR pipelines.
- **Ultralytics YOLO** for object detection.
- **ollama** and the **Qwen (0.5b)** model for language understanding.
- **Raspberry Pi Foundation** for the hardware and community support.
