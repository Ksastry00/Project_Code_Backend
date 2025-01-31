import os
import time
import tempfile
import collections
from io import BytesIO
import threading

import cv2
import numpy as np
from PIL import Image
import torch
from gtts import gTTS
from scipy.io.wavfile import write
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

import gradio as gr
from ultralytics import YOLO
from groq import Groq  # assuming this is a valid library

# Configure PIL as the image backend for consistent image handling
os.environ['ULTRALYTICS_IMAGE_BACKEND'] = 'PIL'

# Load environment variables and initialize Groq client
load_dotenv()
client = Groq()

# Application state
chat_history = []                # Store conversation history
last_frame = None                # Most recent video frame
recording = False                # Audio recording state
detection_history = collections.deque(maxlen=5)  # Recent detections

# Configure GPU acceleration if available
device_yolo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_moondream = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """
    Initialize and configure ML models for object detection and image understanding.
    Returns:
        tuple: (YOLO model for object detection, Moondream model for image captioning/chat)
    """
    # Initialize YOLO (nano model for efficient real-time detection)
    model_yolo = YOLO("yolov10n.pt")
    model_yolo.model.eval()  # set to evaluation mode
    if torch.cuda.is_available():
        model_yolo.model.to(device_yolo)
        try:
            # Convert model to FP16 for faster inference if supported.
            model_yolo.model.half()
        except Exception as e:
            print(f"FP16 conversion not supported for YOLO model: {e}")

    # Load Moondream for image understanding and chat; set to eval mode.
    model_moondream = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": device_moondream}
    )
    model_moondream.eval()

    return model_yolo, model_moondream

# Load models at startup
model_yolo, model_moondream = load_models()

def process_frame(frame):
    """
    Run real-time object detection on a video frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        
    Returns:
        tuple: (detection boxes with coordinates and scores, processed frame as PIL Image)
    """
    global last_frame, detection_history

    # Convert frame to PIL Image once for consistent processing
    last_frame = Image.fromarray(frame)

    # Use no_grad and (if CUDA) autocast for inference speed-up
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                results = model_yolo(frame)
        else:
            results = model_yolo(frame)

    # Extract detection boxes, ensuring they are on CPU for further processing
    boxes = results[0].boxes.data.cpu().numpy()
    return boxes, last_frame

def draw_boxes(frame, boxes):
    """
    Draw bounding boxes and labels onto the image.
    
    Args:
        frame (PIL.Image): The image to annotate
        boxes (numpy.ndarray): Detection boxes with coordinates, confidence, and class index
        
    Returns:
        PIL.Image: Annotated image
    """
    if boxes is None or len(boxes) == 0:
        return frame

    # Get class names from the YOLO model
    class_names = model_yolo.model.names

    # Convert image to NumPy array once for drawing
    frame_np = np.array(frame)
    for *xyxy, conf, cls_idx in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        cls_idx = int(cls_idx)
        class_name = class_names[cls_idx]
        cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_np, f"{class_name} {conf:.2f}", (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return Image.fromarray(frame_np)

def text_to_speech_output(text):
    """
    Convert text to speech (TTS) and return the temporary audio file path.
    
    Args:
        text (str): Text to convert
    
    Returns:
        str or None: Path to generated audio file or None on failure.
    """
    if text:
        try:
            tts = gTTS(text=text, lang='en')
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_file.close()
            tts.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    return None

def transcribe_audio_async(temp_file_name, callback):
    """
    Asynchronously transcribe audio using Groq's Whisper API and generate image captions.
    
    Args:
        temp_file_name (str): Path to temporary audio file
        callback (callable): Function to handle transcription results
    """
    def transcription_thread():
        transcription, caption_text, caption_audio_path = "", "", None
        try:
            with open(temp_file_name, 'rb') as file:
                file_basename = os.path.basename(temp_file_name)
                transcription = client.audio.transcriptions.create(
                    file=(file_basename, file),
                    model="whisper-large-v3",
                    response_format="text",
                    language="en"
                )
                print(f"Received transcription: {transcription}")
                if last_frame:
                    # Use autocast if on CUDA for the caption generation if supported
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                caption_text = model_moondream.caption(last_frame, length="normal")["caption"]
                        else:
                            caption_text = model_moondream.caption(last_frame, length="normal")["caption"]
                    caption_audio_path = text_to_speech_output(caption_text)
                else:
                    caption_text = "No frame available for caption"
        except Exception as e:
            print(f"Groq transcription error: {e}")
            transcription = f"Transcription error: {e}"
            caption_text = "Error during transcription"
        finally:
            # Remove the temporary file
            try:
                os.unlink(temp_file_name)
            except Exception as e:
                print(f"Error cleaning up file: {e}")
            callback(transcription, caption_text, caption_audio_path)
    threading.Thread(target=transcription_thread, daemon=True).start()

def record_audio(audio):
    """
    Process recorded audio and start asynchronous transcription.
    
    Args:
        audio (tuple): (sample_rate, audio_data) from microphone input
        
    Returns:
        tuple: (transcription placeholder, caption placeholder, audio path placeholder)
    """
    transcription_placeholder = "Transcription in progress..."
    caption_placeholder = ""
    caption_audio_placeholder = None

    if audio is not None and isinstance(audio, tuple) and len(audio) == 2:
        try:
            sample_rate, audio_data = audio
            if isinstance(audio_data, np.ndarray):
                # Convert stereo to mono by averaging channels if needed
                if len(audio_data.shape) == 2:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = (audio_data * 32767).astype(np.int16)

                # Write audio to a temporary WAV file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.close()
                write(temp_file.name, sample_rate, audio_data)
                print(f"Saved audio to temporary file: {temp_file.name}")

                # Start asynchronous transcription using a callback to process results
                def transcription_callback(transcription, caption_text, caption_audio_path):
                    print("Final Transcription:", transcription)
                    print("Caption:", caption_text)
                    print("Caption Audio Path:", caption_audio_path)
                    # In production, update shared state or notify UI accordingly.

                transcribe_audio_async(temp_file.name, transcription_callback)
                return transcription_placeholder, caption_placeholder, caption_audio_placeholder

            return "", "No audio data received", None

        except Exception as e:
            print(f"Audio processing error: {e}")
            return f"Error: {e}", "Processing failed", None

    return "", "", None

def chat_with_image(message, history):
    """
    Enable chat interaction regarding the current video frame.
    
    Args:
        message (str): User's query about the image.
        history (list): Conversation history.
        
    Returns:
        tuple: (empty input, updated history, audio response file path)
    """
    if last_frame and message:
        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        response = model_moondream.query(last_frame, message)["answer"]
                else:
                    response = model_moondream.query(last_frame, message)["answer"]
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            audio_path = text_to_speech_output(response)
            return "", history, audio_path
        except Exception as e:
            error_msg = f"Error: {e}"
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            return "", history, None
    return "", history, None

# Build Gradio web interface
with gr.Blocks() as app:
    gr.Markdown("# Real-Time Multimodal Video Analysis")
    
    with gr.Row():
        with gr.Column():
            camera_input = gr.Image(sources=["webcam"], streaming=True)
            video_output = gr.Image(label="Processed Feed", streaming=True)
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="numpy", streaming=False)
            transcription_box = gr.Textbox(label="Transcription")
            caption_box = gr.Textbox(label="Image Caption")
            caption_audio = gr.Audio(label="Caption Audio", format="mp3")
    
    with gr.Row():
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Chat with the model")
        response_audio = gr.Audio(label="Response Audio", format="mp3")
    
    def update_video_feed(frame):
        """
        Process video frames with object detection and return annotated frame.
        """
        boxes, current_frame = process_frame(frame)
        img_with_boxes = draw_boxes(current_frame, boxes)
        return img_with_boxes

    # Set up video stream processing at ~10 FPS.
    camera_input.stream(
        fn=update_video_feed,
        inputs=[camera_input],
        outputs=[video_output],
        queue=True,
        stream_every=0.1
    )
    
    # Set up audio input processing.
    audio_input.change(
        fn=record_audio,
        inputs=[audio_input],
        outputs=[transcription_box, caption_box, caption_audio],
        queue=True
    )
    
    # Set up chat interface.
    msg.submit(
        fn=chat_with_image,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, response_audio],
        queue=True
    )

# Launch Gradio interface with configuration.
app.queue().launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True,
    max_threads=6
)
