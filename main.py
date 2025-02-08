import wave
import numpy as np

from process_audio import process_audio
CHUNK = 1024

audio_path = "audio/exampleaudio.wav"

file_paramaters = tuple()
def main():
    process_audio(audio_path)

main()