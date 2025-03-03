import numpy as np
import matplotlib.pyplot as plt

# Define time range
t = np.linspace(0, 0.02, 1000)  # 20 ms range with fine resolution

# Parameters for the function
A = 1.0  # Amplitudes of oscillations
alpha = 200  # Decay rates
omega = 2000 * np.pi  # Frequencies in rad/s
phi = 0

# Gunshot signal (excluding true impulse)
gunshot_signal = A * np.exp(-alpha * t) * np.sin(omega * t + phi)

# Plot the signal
plt.figure(figsize=(8, 4))
plt.plot(t, gunshot_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Simple Synthetic Gunshot Signal")
plt.grid(True)
plt.legend()
plt.show()
