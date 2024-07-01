import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import os

# Function to transcribe audio in chunks
def transcribe_audio(audio_path, recognizer, chunk_length_ms=60000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = len(audio) // chunk_length_ms + 1
    text = ""

    for i in range(chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, len(audio))
        chunk = audio[start:end]

        chunk_path = f"chunk{i}.wav"
        chunk.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
            try:
                chunk_text = recognizer.recognize_google(audio_data)
                text += chunk_text + " "
            except sr.UnknownValueError:
                print(f"Chunk {i}: Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Chunk {i}: Could not request results from Google Speech Recognition service; {e}")

        os.remove(chunk_path)

    return text

# Load the video file
video_path = "./material_seminar.mp4"
video = mp.VideoFileClip(video_path)

# Extract audio from the video
audio_path = "material_seminar.wav"
video.audio.write_audiofile(audio_path)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Transcribe the audio
transcription = transcribe_audio(audio_path, recognizer)

# Save the transcription to a .md file
md_file_path = "material_seminar.md"
with open(md_file_path, "w") as f:
    f.write(transcription)

print(f"Transcription completed and saved to '{md_file_path}'")






