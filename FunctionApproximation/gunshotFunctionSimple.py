import numpy as np
import matplotlib.pyplot as plt

# 20 ms time range
t = np.linspace(0, 0.02, 1000)
alpha = 200  # Decay rate
omega = 2000 * np.pi  # Frequency

gunshot_signal = np.exp(-alpha * t) * np.sin(omega * t)

plt.figure(figsize=(8, 4))
plt.plot(t, gunshot_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Simple Synthetic Gunshot Signal")
plt.grid(True)
plt.legend()
plt.show()
