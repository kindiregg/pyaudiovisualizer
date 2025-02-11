import sys
import os
import math
import wave
import pygame
import pyaudio
import numpy as np

def get_audio_file_from_folder(folder="audio", extension=".wav"):
    files = [f for f in os.listdir(folder) if f.lower().endswith(extension)]

    if not files:
        raise FileNotFoundError("No wav files found")

    return os.path.join(folder, files[0])

def print_audio_metadata(audio_metadata):
    # For debugging mostly
    print(f"File name: {audio_metadata['file_name']}")
    print(f"Number of channels: {audio_metadata['num_channels']}")
    print(f"Bit depth: {audio_metadata['bit_depth']}")
    print(f"Compression type: {audio_metadata['comp_type']}")
    print(f"Compression name: {audio_metadata['comp_name']}")
    print(f"Sample rate: {audio_metadata['sample_rate']}")
    print(f"Sample length: {audio_metadata['length_mins']} minutes and {audio_metadata['length_seconds_remainder']} seconds")

def get_audio_data(audio_path, audio_metadata):
    song_file = wave.open(audio_path, "rb")
    # Placing all relevant paramaters into a dictionary
    audio_metadata["file_name"] = os.path.basename(audio_path)
    audio_metadata["num_channels"] = song_file.getnchannels()
    audio_metadata["sample_width"] = song_file.getsampwidth()
    audio_metadata["bit_depth"] = audio_metadata["sample_width"] * 8
    audio_metadata["comp_type"] = song_file.getcomptype()
    audio_metadata["comp_name"] = song_file.getcompname()
    audio_metadata["sample_rate"] = song_file.getframerate()
    audio_metadata["lengh_samples"] = song_file.getnframes()
    audio_metadata["length_time"] = audio_metadata["lengh_samples"] / audio_metadata["sample_rate"] / 60
    audio_metadata["length_mins"] = int(audio_metadata["length_time"])
    audio_metadata["length_seconds_remainder"] = int((audio_metadata["length_time"] - audio_metadata["length_mins"]) * 60)
    song_file.close()
    print_audio_metadata(audio_metadata)
    return audio_metadata

def decode_24bit_to_int32(raw_bytes):
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_samples = len(byte_array) // 3
    byte_array = byte_array[:n_samples * 3].reshape(n_samples, 3)
    
    # Combine the 3 bytes into a 32-bit integer.
    # For little-endian, the sample is:
    # sample = byte0 + (byte1 << 8) + (byte2 << 16)
    samples_uint32 = (
        byte_array[:, 0].astype(np.uint32) |
        (byte_array[:, 1].astype(np.uint32) << 8) |
        (byte_array[:, 2].astype(np.uint32) << 16)
    )
    
    # Convert to signed 32-bit integers.
    samples = samples_uint32.astype(np.int32)
    
    # Sign-extend from 24 bits: if the 24th bit is set, subtract 0x1000000.
    mask = 0x800000  # 1 << 23; this is the sign bit in 24-bit data.
    samples[samples & mask != 0] -= 0x1000000
    
    return samples

def convert_bytes_to_numpy_array(raw_bytes, sample_width):

    match sample_width:
        case 1:
            samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        case 2:
            samples = np.frombuffer(raw_bytes, dtype=np.int16)
        case 3:
            # insert long winded thing here
            samples = decode_24bit_to_int32(raw_bytes)
        case 4:
            samples = np.frombuffer(raw_bytes, dtype=np.float32)
        case _:
            raise ValueError("Unsupported bit depth")

    return samples

def read_wav_in_chunks(audio_path, chunk_size=1024, audio_metadata=None):
    if audio_metadata is None:
        raise Exception("Unknown error, audio metadata is missing")

    with wave.open(audio_path, "rb") as song_file:
        while True:
            raw_bytes = song_file.readframes(chunk_size)
            if not raw_bytes:
                break
            samples = convert_bytes_to_numpy_array(raw_bytes, audio_metadata["sample_width"])

            if audio_metadata["num_channels"] > 1:
                samples = samples.reshape(-1, audio_metadata["num_channels"])

            yield samples

def get_audio_chunks(audio_path, chunk_size=1024, audio_metadata=None):
    chunks = []
    for chunk in read_wav_in_chunks(audio_path, chunk_size, audio_metadata):
        chunks.append(chunk)
    print(f"Got {len(chunks)} chunks")
    return chunks

def array_to_fft(chunks):
    for i, chunk in enumerate(chunks):
        chunk = chunk[:, 0] if chunk.ndim > 1 else chunk
        fft_vals = np.fft.fft(chunk)
        fft_mag = np.abs(fft_vals)
        N = len(chunk)
        half = N // 2
        fft_mag = fft_mag[:half]
        yield fft_mag

def build_freq_bands(center_freqs):
    # Given a center frequency, return the lower and upper bounds band
    bands = []
    num_bands = len(center_freqs)
    ratio = 2 ** (1/num_bands)
    # We're letting the bands' scope overlap on purpose because it looks better in especially the lower frequencies
    for center in center_freqs:
        low_edge = center / ratio
        high_edge = center * ratio
        bands.append((low_edge, high_edge))
    return bands

def bin_fft_to_bands(fft_freqs, fft_mag, bands):
    """
    :param fft_freqs: array of frequencies for each bin (same length as fft_mag)
    :param fft_mag: array of magnitudes (e.g. np.abs(rfft(audio_chunk)))
    :param bands: list of (low_edge, high_edge) for each 1/3-oct band
    :return: list of float values (averages) for each band
    """
    binned = []
    for (low, high) in bands:
        mask = (fft_freqs >= low) & (fft_freqs < high)
        if np.any(mask):
            binned.append(np.mean(fft_mag[mask]))
        else:
            binned.append(0.0)
    return binned

def adjust_height_by_freq(screen, band_mags):
    # Roughly adjusting the height of the bar by the frequency it represents, the lower being lower
    width, height = screen.get_size()
    band_mags = np.array(band_mags, dtype=float)
    adjusted_mags = band_mags.copy()
    num_bands = len(band_mags)
    max_val = max(adjusted_mags) if num_bands > 0 else 1e-12
    bar_width = width / num_bands
    for i, val in enumerate(adjusted_mags):
        bar_height = max((val / max_val) * (height * 0.9), 1-(1e-12))
        x_pos = i * bar_width
        relative_x_pos = x_pos / width
        if i < 4 or i > num_bands - 8:
            # boosting low end a tiny bit if they are actually playing
            adjusted_mags[i] = bar_height + (relative_x_pos ** 1.5)
        if i > num_bands - 6:
            # boosting the top end a little as well to account for roll-off
            adjusted_mags[i] = bar_height + (relative_x_pos ** 1.1)
        new_height = bar_height - math.log2(relative_x_pos + 1e-12)
        if new_height <= 0:
            new_height = 1e-12
        adjusted_mags[i] = new_height
    return adjusted_mags
    

def draw_binned_bars(screen, band_mags, color=(0, 200, 255)):
    
    width, height = screen.get_size()
    num_bands = len(band_mags)
    max_val = max(band_mags) if num_bands > 0 else 1e-12
    bar_width = width / num_bands

    for i, val in enumerate(band_mags):
        bar_height = max((val / max_val) * (height * 0.9), 1-(1e-12))
        x_pos = i * bar_width
        y_pos = height - bar_height
        if math.isnan(y_pos):
            y_pos = 1e-12
            # I cant be bothered to figure out why this happens rn lol
        rect_x = int(x_pos)
        rect_y = int(y_pos)
        rect_w = max(int(bar_width), 1)
        rect_h = max(int(bar_height), 1)

        pygame.draw.rect(screen, color, (rect_x, rect_y, rect_w, rect_h))

def adjust_band_heights_local_relative(band_mags, window=5, decay=0.8, threshold=0.5):
    # Smooth the band magnitudes by computing a moving average,
    # and then take the elementwise maximum of the original and smoothed values.
    # This helps fill in bands that might be zero because in the lower frequencies where information is often sparse.
    # i.e. a note at 30hz has a first harmonic at 60hz

    band_mags = np.array(band_mags, dtype=float)
    adjusted_mags = band_mags.copy()
    num_mags = len(band_mags)
    
    for i in range(num_mags):
        # local window for ajustement
        start = max(0, i - window)
        end = min(num_mags, i + window + 1)
        local_max = np.max(band_mags[start:end])

        if local_max > 0 and band_mags[i] < threshold * local_max:
            adjusted_mags[i] = decay * local_max
            ajustable_distance = [
                abs(i - j)
                for j in range(start, end)
                if band_mags[j] >= 0.9 * local_max
            ]
            if ajustable_distance:
                closest = min(ajustable_distance)
                adjusted_mags[i] = band_mags[i] + (local_max - band_mags[i]) * (decay ** closest)

    return adjusted_mags

def update_display_values(display, new_vals, attack_amount=0.8, decay_amount=0.2):
    current_display = np.array(display, dtype=float)
    new_vals = np.array(new_vals, dtype=float)
    ratio = new_vals / (current_display + 1e-12)

    updated_vals = np.where(new_vals > current_display,
                   current_display * (ratio ** attack_amount),
                   current_display * (ratio ** decay_amount))

    return updated_vals

def play_audio_and_visualize(
    audio_metadata=None,
    audio_path=None,
    screen_size=(800, 600)
):
    if audio_path is None:
        raise Exception("No audio path provided")
    
    audio_chunks = get_audio_chunks(audio_path, CHUNK, audio_metadata)
    if not audio_chunks:
        raise Exception("No audio found")
    
    fft_generator = array_to_fft(audio_chunks)
    sample_rate = audio_metadata["sample_rate"]

    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(audio_metadata["file_name"])
    clock = pygame.time.Clock()

    wf = wave.open(audio_path, "rb")
    
    p = pyaudio.PyAudio()

    if sys.platform.startswith('win'):
        output_device_index = None
        # WSL btw
    else:
        output_device_index = None
    
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
        output_device_index=output_device_index,
        frames_per_buffer=CHUNK
    )
    
    running = True
    bands = build_freq_bands(BAND_CENTERS)
    M = len(bands)
    current_vals = np.full(M, 1e-12)
    # ------------------------------
    # --- This is the main loop ---
    # ------------------------------
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        audio_data = wf.readframes(CHUNK)
        if not audio_data:
            running = False

        if not running:
            break

        stream.write(audio_data)
        
        screen.fill((0, 0, 0))


        try:
            fft_mag = next(fft_generator)
        except StopIteration:
            # End of data
            running = False

        num_bins = len(fft_mag)
        fft_freqs = np.fft.rfftfreq(2 * (num_bins - 1), d=1.0/sample_rate)
        band_mags = bin_fft_to_bands(fft_freqs, fft_mag, bands)
        band_mags_freq_adjusted = adjust_height_by_freq(screen, band_mags)
        band_mags_localized = adjust_band_heights_local_relative(band_mags_freq_adjusted, window=9, decay=0.6, threshold=0.5)
        atk = 0.8
        current_vals = update_display_values(current_vals, band_mags_localized, attack_amount=atk, decay_amount=(atk/4))
        draw_binned_bars(screen, current_vals)
        pygame.display.flip()
        clock.tick(60)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    print("Reading wav file...")
    global CHUNK
    CHUNK = 2048
    # from get_bands.py because I didn't want to generate the bands every time I ran
    global BAND_CENTERS
    BAND_CENTERS =[31, 32, 34, 37, 39, 41, 44, 47, 50, 53, 56, 60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 129, 140, 151, 163, 175, 189, 204, 221, 238, 257, 277, 300, 323, 349, 377, 407, 439, 474, 511, 552, 596, 643, 694, 749, 809, 873, 942, 1017, 1097, 1185, 1279, 1380, 1490, 1608, 1736, 1873, 2022, 2183, 2356, 2543, 2745, 2962, 3198, 3451, 3725, 4021, 4340, 4685, 5057, 5458, 5892, 6359, 6864, 7409, 7997, 8632, 9317, 10057, 10855, 11717, 12647, 13651, 14734, 15904, 17166, 18529, 20000]
    audio_path = get_audio_file_from_folder()
    audio_metadata = get_audio_data(audio_path, dict())
        
    play_audio_and_visualize(audio_metadata, audio_path, screen_size = (1920, 1080))
