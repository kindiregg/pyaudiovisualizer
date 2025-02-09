import numpy as np

from process_audio import get_audio_data, get_audio_chunks
CHUNK = 1024

audio_path = "audio/Eastghost - Sunshine (Kindrid Bootleg) [16bit].wav"
audio_metadata = {}

def main():
    print("Reading wav file...")
    get_audio_data(audio_path, audio_metadata)
    audio_chunks = get_audio_chunks(audio_path, CHUNK, audio_metadata)
    # print(f"Audio chunks: {audio_chunks}")
                                        
main()