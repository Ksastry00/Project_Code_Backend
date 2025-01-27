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
import collections
from IPython.display import Audio
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForVision2Seq, YolosImageProcessor, YolosForObjectDetection
from transformers.image_utils import load_image
import torch
import requests

load_dotenv()
client = Groq()

# Global state
chat_history = []
last_frame = None
recording = False
detection_history = collections.deque(maxlen=5)

# Device configuration
device_yolo = torch.device("cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model_yolos = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(DEVICE)

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model_smolvlm = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

def process_frame(frame):
    global last_frame, detection_history
    
    # Store the RGB version for the model
    last_frame = Image.fromarray(frame)
    
    # Object detection using YOLOS
    pil_image = Image.fromarray(frame)
    inputs = image_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    outputs = model_yolos(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(DEVICE)
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    
    img_with_boxes = np.array(pil_image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        
        cv2.rectangle(img_with_boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img_with_boxes, f"{model_yolos.config.id2label[label.item()]} {round(score.item(), 3)}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes

tts_temp_file = None
def text_to_speech_output(text):
    global tts_temp_file
    if text:
        try:
            tts = gTTS(text=text, lang='en')
            if tts_temp_file is None:
                tts_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts_temp_file.close()
            tts.save(tts_temp_file.name)
            
            # Return just the file path for Gradio audio component
            return tts_temp_file.name
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None
    return None

transcription_cache = {}
def record_audio(audio):
    global transcription_cache
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
                if temp_file.name in transcription_cache:
                    transcription = transcription_cache[temp_file.name]
                else:
                    with open(temp_file.name, 'rb') as file:
                        transcription = client.audio.transcriptions.create(
                            file=(os.path.basename(temp_file.name), file),
                            model="whisper-large-v3-turbo",
                            response_format="json",
                            language="en",
                        )
                    transcription_cache[temp_file.name] = transcription
                
                # Generate caption and TTS
                if last_frame:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": transcription.text},
                            ]
                        },
                    ]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=[last_frame], return_tensors="pt")
                    inputs = inputs.to(DEVICE)

                    generated_ids = model_smolvlm.generate(**inputs, max_new_tokens=500)
                    generated_texts = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    caption = generated_texts[0]
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": message}
                    ]
                },
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[last_frame], return_tensors="pt")
            inputs = inputs.to(DEVICE)

            generated_ids = model_smolvlm.generate(**inputs, max_new_tokens=500)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            response = generated_texts[0]
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
with gr.Blocks(delete_cache=(60, 60)) as app:
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