import gradio as gr
import torch
from PIL import Image
import tempfile
import os
import numpy as np
from io import BytesIO
import base64
from groq import Groq
from dotenv import load_dotenv
import cv2
import time
from gtts import gTTS
import sounddevice as sd
from scipy.io.wavfile import write
import torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import collections
from IPython.display import Audio

load_dotenv()
client = Groq()

# Global state
chat_history = []
last_frame = None
recording = False
detection_history = collections.deque(maxlen=5)

# Device configuration
device_yolo = torch.device("cpu")
device_moondream = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    model_yolo = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device_yolo)
    model_yolo.eval()

    model_moondream = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": device_moondream}
    )
    return model_yolo, model_moondream

model_yolo, model_moondream = load_models()

def process_frame(frame):
    global last_frame, detection_history
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    last_frame = Image.fromarray(frame_rgb)
    
    # Object detection
    with torch.no_grad():
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device_yolo)
        predictions = model_yolo(img_tensor)[0]

    boxes = predictions['boxes'].cpu()
    scores = predictions['scores'].cpu()
    labels = predictions['labels'].cpu()
    
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    
    detected_objects = []
    img_with_boxes = np.copy(frame_rgb)
    
    for i in keep:
        if scores[i] > 0.7:
            box = boxes[i].int().tolist()
            label = labels[i].item()
            score = scores[i].item()
            class_name = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"][label]
            detected_objects.append(class_name)
            
            cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"{class_name}: {score:.2f}", 
                       (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    detection_history.append(detected_objects)
    return img_with_boxes

def text_to_speech_output(text):
    if text:
        try:
            tts = gTTS(text=text, lang='en')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.close()
            tts.save(temp_file.name)
            
            # Return just the file path for Gradio audio component
            return temp_file.name
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None
    return None

def record_audio(audio):
    temp_file = None
    if audio is not None and isinstance(audio, tuple) and len(audio) == 2:
        try:
            sample_rate, audio_data = audio
            
            if isinstance(audio_data, np.ndarray):
                # Ensure audio data is in correct format
                if len(audio_data.shape) == 2:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Save to temporary WAV file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.close()
                write(temp_file.name, sample_rate, audio_data)
                
                # Transcribe with Groq
                with open(temp_file.name, 'rb') as file:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(temp_file.name), file),
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en",
                    )
                
                # Generate caption and TTS
                if last_frame:
                    caption = model_moondream.caption(last_frame, length="normal")["caption"]
                    audio_path = text_to_speech_output(caption)
                    return transcription.text, caption, audio_path
            
            return "", "No audio data received", None
            
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return f"Error: {str(e)}", "Processing failed", None
            
        finally:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    print(f"Error cleaning up file: {str(e)}")
    
    return "", "", None

def chat_with_image(message, history):
    if last_frame and message:
        try:
            response = model_moondream.query(last_frame, message)["answer"]
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            audio_path = text_to_speech_output(response)
            return "", history, audio_path
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            return "", history, None
    return "", history, None

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Real-Time Multimodal Video Analysis")
    
    with gr.Row():
        with gr.Column():
            camera_input = gr.Image(sources=["webcam"], streaming=True)  # Changed 'source' to 'sources'
            video_output = gr.Image(label="Processed Feed")
        
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", streaming=False)  # Also updated Audio component
            transcription = gr.Textbox(label="Transcription")
            caption = gr.Textbox(label="Image Caption")
            caption_audio = gr.Audio(label="Caption Audio", format="mp3")
    
    with gr.Row():
        chatbot = gr.Chatbot(type="messages")  # Changed to use messages format
        msg = gr.Textbox(label="Chat with the model")
        response_audio = gr.Audio(label="Response Audio", format="mp3", autoplay=True)  # Added autoplay

    # Set up event handlers
    camera_input.stream(process_frame, inputs=[camera_input], outputs=[video_output])
    audio_input.stop_recording(record_audio, inputs=[audio_input], 
                             outputs=[transcription, caption, caption_audio])
    msg.submit(chat_with_image, inputs=[msg, chatbot], 
              outputs=[msg, chatbot, response_audio])

# Launch the app with corrected parameters
app.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=7860)