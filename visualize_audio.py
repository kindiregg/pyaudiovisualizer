import sys
import math
import pygame
import numpy as np

def get_display_bar_width(freq, screen_size):
    width, _ = screen_size
    min_width = 10
    peak_width =  min(int(width / 200), min_width)
    return peak_width * (1 + (1/freq)) ** 40
    


def visualize_fft(
    fft_generator,
    sample_rate,
    screen_size=(800, 600),
    use_log2=True
):
    """
    Visualize FFT magnitude data from 'fft_generator' in Pygame.

    :param fft_generator: An iterator (or generator) that yields 1D NumPy arrays of FFT values (complex or already magnitudes).
    :param sample_rate: The sample rate of your audio (used for frequency calculations).
    :param screen_size: (width, height) tuple for the Pygame window.
    :param use_log2: If True, position bins using log2(frequency). If False, use a linear bin index (simple bar for each FFT bin).
    """
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Real-Time FFT Visualization")

    clock = pygame.time.Clock()
    width, height = screen_size

    bar_color = (0, 200, 255)
    running = True

    for fft_data in fft_generator:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        screen.fill((0, 0, 0))
        mag = np.abs(fft_data)
        max_val = np.max(mag) if mag.size > 0 else 1e-12
        if max_val < 1e-12:
            max_val = 1.0

        # 6. Draw each bin as a rectangle
        num_bins = len(mag)
        if num_bins == 0:
            # No data, skip this frame
            pygame.display.flip()
            clock.tick(30)
            continue

        if use_log2:
            # --- Log2 frequency positioning approach ---
            freq_indices = np.arange(1, num_bins)
            # freqs = freq_indices * (sample_rate / (2.0 * num_bins))

            max_freq = sample_rate / 2
            a_constant = width / math.log2(max_freq + 1) if max_freq > 0 else 1
            
            for i in freq_indices:
                # Compute frequency
                freq = i * (sample_rate / (2.0 * num_bins))
                x_pos = a_constant * math.log2(freq + 1)
                
                # bar height
                bar_height = (mag[i] / max_val) * (height * 0.9)
                rect_x = int(x_pos)
                rect_y = int(height - bar_height)
                rect_w = get_display_bar_width(freq, screen_size)
                rect_h = max(int(bar_height), 1)

                pygame.draw.rect(
                    screen,
                    bar_color,
                    pygame.Rect(rect_x, rect_y, rect_w, rect_h)
                )

        else:
            # --- Simple linear approach: each bin side by side ---
            bar_width = width / num_bins

            for i in range(num_bins):
                bar_height = (mag[i] / max_val) * (height * 0.9)
                x_pos = i * bar_width
                y_pos = height - bar_height

                rect_x = int(x_pos)
                rect_y = int(y_pos)
                rect_w = max(int(bar_width), 1)
                rect_h = max(int(bar_height), 1)

                pygame.draw.rect(
                    screen,
                    bar_color,
                    pygame.Rect(rect_x, rect_y, rect_w, rect_h)
                )

        # 7. Flip the display to show the updated screen
        pygame.display.flip()

        # 8. Limit FPS (e.g. 30)
        clock.tick(30)

    pygame.quit()
    sys.exit()
