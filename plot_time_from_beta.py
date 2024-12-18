import matplotlib.pyplot as plt
from pi_zero_pytorch.pi_zero import default_sample_times

times = default_sample_times((10000,), s = 0.9)

plt.hist(times.numpy())
plt.show()
