from gtts import gTTS
import os

def text_to_speech(text, lang='en', filename='output.mp3'):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f"Saved speech to {filename}")

if __name__ == "__main__":
    text = "Hello, this is a demo of gTTS."
    text_to_speech(text)
