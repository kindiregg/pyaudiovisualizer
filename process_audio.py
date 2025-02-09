import wave
import numpy as np

def print_audio_metadata(audio_metadata):
    # For debugging mostly
    print(f"Number of channels: {audio_metadata['num_channels']}")
    print(f"Bit depth: {audio_metadata['bit_depth']}")
    print(f"Compression type: {audio_metadata['comp_type']}")
    print(f"Compression name: {audio_metadata['comp_name']}")
    print(f"Sample rate: {audio_metadata['sample_rate']}")
    print(f"Sample length: {audio_metadata['length_mins']} minutes and {audio_metadata['length_seconds_remainder']} seconds")

def get_audio_data(audio_path, audio_metadata):
    song_file = wave.open(audio_path, "rb")
    # Placing all relevant paramaters into a dictionary
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

def decode_24bit_to_int32(raw_bytes):
    # This is a placeholder for now
    # TODO: implement this for 24 bit audio 
    raise NotImplementedError("24 bit audio not supported yet")

def convert_bytes_to_numpy_array(raw_bytes, sample_width):
    # convert raw WAV framse to a NumPy array based on given sample width
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
        case _: # default
            raise ValueError("Unsupported bit depth")

    # print(f"Samples: {samples}")
    return samples

def read_wav_in_chunks(audio_path, chunk_size=1024, audio_metadata=None):
    if audio_metadata is None:
        raise Exception("Unknown error, audio metadata is missing")
    if type(audio_metadata) is not dict:
        raise Exception("Unknown error, audio metadata is not a dictionary")
    print(f"metadata type: {type(audio_metadata)}")
    with wave.open(audio_path, "rb") as song_file:
        # i = 0
        # i for debugging purposes
        while True:
            raw_bytes = song_file.readframes(chunk_size)
            if not raw_bytes:
                break
            samples = convert_bytes_to_numpy_array(raw_bytes, audio_metadata["sample_width"])

            if audio_metadata["num_channels"] > 1:
                samples = samples.reshape(-1, audio_metadata["num_channels"])

            # print(f"Chunk {i}: {samples}")
            # i += 1
            yield samples

def get_audio_chunks(audio_path, chunk_size=1024, audio_metadata=None):
    chunks = []
    for i, chunk in enumerate(read_wav_in_chunks(audio_path, chunk_size, audio_metadata)):
        print(f"Got chunk {i}, shape: {chunk.shape}")
        chunks.append(chunk)
    return chunks