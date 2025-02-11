import numpy as np

# I used this to generate a set number of frequency bands for the FFT becuase I didn't want to generate them every single time.
# And because I changed some values manually lol

def generate_log_bands(total_bands):
    f_min = 65 #hz
    f_max = 20000 #hz
    num_bands = total_bands - 12
    points = np.logspace(np.log10(f_min), np.log10(f_max), num=num_bands)
    return np.round(points).astype(int).tolist()
def get_bands(total_bands):
    upper_bands = generate_log_bands(total_bands)
    # The sub range was giving me trouble so I just hardcoded it
    # Approx Freq Values of int B0 to B1
    lower_bands = [31, 32, 34, 37, 39, 41, 44, 47, 50, 53, 56, 60]
    bands = lower_bands + upper_bands
    return bands

bands = get_bands(88)
print(bands)
