import numpy as np
import matplotlib.pyplot as plt

# This code needs to be redone in depth in order to mimic what we can see on the gunshot_signal.pdf

# Time range
t = np.linspace(0, 0.2, 1000)  # 200 ms range with fine resolution

# Shockwave (Mach wave travelling at bullet speed) is heard from 10 to 20ms on gunshot_signal.pdf recording
delta1 = 0.01

# Muzzle blast (Wave from gas ignition travelling at near-sonic speed) is heard from 150 to 160ms on gunshot_signal.pdf recording
delta2 = 0.15

# Amplitudes of oscillations
A1, A2 = 1.0, 0.6

# Decay rates
alpha1, alpha2 = 200, 200

# Frequencies in rad/s
omega1, omega2 = 1500 * np.pi, 500 * np.pi

# Define the damped oscillations
oscillation1 = np.where(
    t >= delta1, A1 * np.exp(-alpha1 * (t - delta1)) * np.sin(omega1 * (t - delta1)), 0
)
oscillation2 = np.where(
    t >= delta2, A2 * np.exp(-alpha2 * (t - delta2)) * np.sin(omega2 * (t - delta2)), 0
)

# Gunshot signal (excluding true impulse)
gunshot_signal = oscillation1 + oscillation2

# Plot the signal
plt.figure(figsize=(8, 4))
plt.plot(t, gunshot_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Synthetic Gunshot Signal")
plt.grid(True)
plt.legend()
plt.show()
