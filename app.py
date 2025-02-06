import os
import base64
import tempfile
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import torch
from gtts import gTTS
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from groq import Groq
from dotenv import load_dotenv
from scipy.io.wavfile import write
import asyncio

load_dotenv()
app = Flask(__name__)
client = Groq()

# Initialize YOLO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov10n.pt")
model.model.eval()

if torch.cuda.is_available():
    model.model.to(device)
    try:
        model.model.half()
    except Exception as e:
        print(f"FP16 conversion not supported: {e}")

latest_frame = None
latest_detections = None

def process_image(frame):
    global latest_detections
    try:
        if isinstance(frame, Image.Image) and frame.mode != 'RGB':
            frame = frame.convert('RGB')
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    results = model(frame, conf=0.35)
            else:
                results = model(frame, conf=0.35)
        latest_detections = results[0].boxes.data.cpu().numpy()
        return latest_detections
    except Exception as e:
        print(f"Error in process_image: {e}")
        return latest_detections if latest_detections is not None else np.array([])

def draw_boxes(image, boxes):
    try:
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for *xyxy, conf, cls_idx in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_idx = int(cls_idx)
            class_name = model.model.names[cls_idx]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_name} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_np, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         (0, 255, 0),
                         -1)
            cv2.putText(img_np, text,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error in draw_boxes: {e}")
        return image

def analyze_with_groq(image_frame, detected_objects, query=None):
    try:
        img_io = BytesIO()
        image_frame.save(img_io, format='JPEG', quality=95)
        img_io.seek(0)
        base64_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        objects_context = ""
        if detected_objects:
            objects_context = f"Objects detected in the scene: {', '.join(detected_objects)}. "

        user_prompt = query if query else "What do you see in this image? Be concise."
        prompt = f"{objects_context}{user_prompt}"

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.7,
            max_completion_tokens=128,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in analyze_with_groq: {str(e)}")
        return "Error analyzing image"

def text_to_speech(text):
    if not text:
        return None
    try:
        # Split text into shorter phrases for faster speech
        phrases = [p.strip() for p in text.replace('.', '.|').replace('!', '!|').replace('?', '?|').split('|') if p.strip()]
        audio_data = BytesIO()
        
        for phrase in phrases:
            tts = gTTS(text=phrase, lang='en', slow=False)
            tts.write_to_fp(audio_data)
        
        audio_data.seek(0)
        encoded_data = base64.b64encode(audio_data.getvalue()).decode()
        return f"data:audio/mp3;base64,{encoded_data}"
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        global latest_frame
        image_file = request.files['image']
        if not image_file.content_type.startswith('image/'):
            return jsonify({'error': 'Invalid image format'}), 400
            
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        latest_frame = image.copy()
        boxes = process_image(image)
        
        if len(boxes) > 0:
            annotated_image = draw_boxes(image, boxes)
        else:
            annotated_image = image
        
        img_io = BytesIO()
        annotated_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        img_str = base64.b64encode(img_io.getvalue()).decode()
        
        return jsonify({
            'image': f'data:image/jpeg;base64,{img_str}',
        })
        
    except Exception as e:
        print(f"Error in /analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        if latest_frame is None:
            return jsonify({'error': 'No image available'}), 400
            
        boxes = process_image(latest_frame)
        detected_objects = list(set([model.model.names[int(box[5])] for box in boxes]))
        response = analyze_with_groq(latest_frame, detected_objects, data['message'])
        audio_data = text_to_speech(response)
        
        return jsonify({
            'response': response,
            'audio': audio_data
        })
        
    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': str(e)}), 500

def transcribe_audio(temp_file_name):
    """
    Transcribe audio using Groq's Whisper API
    """
    try:
        with open(temp_file_name, 'rb') as file:
            file_basename = os.path.basename(temp_file_name)
            transcription = client.audio.transcriptions.create(
                file=(file_basename, file),
                model="distil-whisper-large-v3-en",
                response_format="text",
                language="en"
            )
            print(f"Received transcription: {transcription}")
            return transcription
    except Exception as e:
        print(f"Groq transcription error: {e}")
        return f"Transcription error: {e}"
    finally:
        try:
            os.unlink(temp_file_name)
        except Exception as e:
            print(f"Error cleaning up file: {e}")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    try:
        audio_file = request.files['audio']
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
        temp_file.close()
        audio_file.save(temp_file.name)
        
        transcription = transcribe_audio(temp_file.name)
        
        return jsonify({
            'text': transcription.strip() if isinstance(transcription, str) else str(transcription)
        })
        
    except Exception as e:
        print(f"Error in /transcribe: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
