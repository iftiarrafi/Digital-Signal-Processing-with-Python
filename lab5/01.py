import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 1, 1000)  # 1 second duration, 1000 points
A = 1  # Amplitude

frequencies = [1, 3, 5, 7, 9]  # Different frequencies
resultant_x = np.zeros_like(t)  # Initialize resultant x with zeros

# Generate and add sine waves
for f in frequencies:
    x = A * np.sin(2 * np.pi * f * t)
    resultant_x += x

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(t, resultant_x, label="Sum of sinusoids")
plt.title("Sum of Multiple Sine Waves")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
