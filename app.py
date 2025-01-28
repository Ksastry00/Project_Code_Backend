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
from transformers import AutoModelForCausalLM, AutoTokenizer
import collections
from IPython.display import Audio
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

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
    model_yolo = YOLO("yolov10n.pt")
    
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
    
    # Store the RGB version for the model
    last_frame = Image.fromarray(frame)
    
    results = model_yolo(frame)
    img_with_boxes = results[0].plot()
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
            camera_input = gr.Image(sources=["webcam"], streaming=True)
            video_output = gr.Image(label="Processed Feed", streaming=True)
        
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy")
            transcription = gr.Textbox(label="Transcription")
            caption = gr.Textbox(label="Image Caption")
            caption_audio = gr.Audio(label="Caption Audio", format="mp3")
    
    with gr.Row():
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Chat with the model")
        response_audio = gr.Audio(label="Response Audio", format="mp3")

    # Set up event handlers
    camera_input.stream(
        fn=process_frame,
        inputs=[camera_input],
        outputs=[video_output],
        queue=True,
        stream_every=0.1  # 10 FPS
    )
    
    audio_input.stop_recording(
        fn=record_audio,
        inputs=[audio_input],
        outputs=[transcription, caption, caption_audio],
        queue=True
    )
    
    msg.submit(
        fn=chat_with_image,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, response_audio],
        queue=True
    )

app.queue().launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True,
    max_threads=3
)