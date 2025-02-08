import wave
import numpy as np

def process_audio(audio_path):
    print("Reading wav file...")
    song_file = wave.open(audio_path, "rb")
    # print(song_file.readframes(10)) 
    nchannels = song_file.getnchannels()
    print(f"Number of channels: {nchannels}")
    bwitdh = song_file.getsampwidth()
    print(f"Sample width: {bwitdh}")
    bdepth = song_file.getsampwidth() * 8
    print(f"Bit depth: {bdepth}")
    comp_type = song_file.getcomptype()
    print(f"Compression type: {comp_type}")
    comp_name = song_file.getcompname()
    print(f"Compression name: {comp_name}")
    sample_rate = song_file.getframerate()
    print(f"Sample rate: {sample_rate}")
    length = song_file.getnframes()
    length_time = length / sample_rate / 60
    length_mins = int(length_time)
    length_seconds_remainder = int((length_time - length_mins) * 60)
    print(f"Length: {length_mins} minutes {length_seconds_remainder} seconds")
    song_file.close()

# TODO: numpy array of audio data