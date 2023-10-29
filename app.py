from flask import Flask, Response, jsonify
import cv2
import numpy as np
import pyaudio
import wave
import openai
from gtts import gTTS
import pygame
import os
from dotenv import load_dotenv

app = Flask(__name__)

def generate_frames():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture(1)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 480))
        boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05, useMeanshiftGrouping=False)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Step 1: Play the help.wav audio
    pygame.mixer.init()
    pygame.mixer.music.load("help.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    # Step 2: Stream and record audio for 5 seconds
    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        audio_data = stream.read(CHUNK)
        frames.append(audio_data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Step 3: Transcribe the recorded audio
    audio_file = open("output.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)["text"]

    # Step 4: Generate GPT-3 response
    pre_prompt = f'''Act as a relief SPOT robot for a human in a disaster scenario Human:  {transcript}. reply in the language of the human. 

    reassure the human that help is on its way, and that they are safe. state that images, voice, and geolocation data is being sent in real time to the rescue teams. 

    just return the text to be spoken by the robot in the language of the human. do not return any other data. do not repeat the human input. just reply to it using above instructions
    '''
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=pre_prompt,
        max_tokens=100
    )
    generation = response.choices[0].text.strip()

    # Step 5: Speak out the GPT-3 response
    tts = gTTS(text=generation, lang='en')
    tts.save("gpt_response.mp3")
    pygame.mixer.music.load("gpt_response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    return jsonify({'response': 'Process completed successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
