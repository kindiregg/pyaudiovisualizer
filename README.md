# PyAudioVisualize

I made a very scrappy version of the [Monstercat Visualizer](https://www.youtube.com/watch?v=PKfxmFU3lWY) in Python wao very cool!
Place your wav file in the audio folder and run!
I included a [bootleg remix I made](https://soundcloud.com/kindridmusic/eastghost-sunshine-kindrid-bootleg) for testing :smile:.

## Features

- **Real-Time Audio Playback:** Plays a WAV file using PyAudio.
- **FFT Analysis:** Processes audio in chunks to compute FFT magnitudes.
- **Frequency Binning:** Supports binning FFT data into standard 1/3-octave bands or custom log-spaced bands.
- **Smooth Visualization:** Uses attack/decay smoothing (either additive or multiplicative) so that the visual bars “rise” quickly and decay slowly.
- **Cross-Platform:** Designed to run on Windows and Linux (note: if using WSL, additional audio configuration may be required).
- **Customizable:** Easily adjust chunk size, band definitions, smoothing parameters, and more.

## Requirements

- Python 3.6+
- [numpy](https://numpy.org/)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [Pygame](https://www.pygame.org/)
> You can also run pip install -r requirements.txt
> It is reccomended to run this in a [virtual environment](https://docs.python.org/3/library/venv.html) to avoid messing up your global installs.

> **Note for Windows Users:**  
> PyAudio installation may require precompiled wheels. Check [this repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/) if you have trouble installing.
> Does not work in WSL due to audio driver limitations

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kindiregg/pyaudiovisualizer.git
   cd pyaudiovisualize
