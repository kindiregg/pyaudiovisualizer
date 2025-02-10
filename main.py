import pyaudio
import wave

from process_audio import get_audio_data, get_audio_chunks, array_to_fft
from visualize_audio import visualize_fft

CHUNK = 2048
audio_path = "audio/Eastghost - Sunshine (Kindrid Bootleg) [16bit].wav"
screen_size = (1920, 1080)

def play_audio(audio_path):
    p = pyaudio.PyAudio()
    wf = wave.open(audio_path, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    print("Reading wav file...")
    audio_metadata = get_audio_data(audio_path, dict())
    audio_chunks = get_audio_chunks(audio_path, CHUNK, audio_metadata)
    # print(f"Audio chunks 1: {audio_chunks[0]}")
    fft_gen = array_to_fft(audio_chunks)
    # play_audio(audio_path)
    visualize_fft(fft_gen, audio_metadata["sample_rate"],screen_size, use_log2=True)
    # print(f"FFT chunks 1: {next(fft_chunks)}")
                                        
main()