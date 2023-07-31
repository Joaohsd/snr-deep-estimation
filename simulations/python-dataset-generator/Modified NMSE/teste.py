import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N = 100000000

noise_1 = np.random.randn(N)
noise_2 = np.random.randn(N)

noise = noise_1 + noise_2 *1j

data = {
    "noise_1": noise_1,
    "noise_2": noise_2
}

df = pd.DataFrame(data)

print(df.corr())

fig = plt.figure(figsize=(8,8))
plt.plot(np.real(noise), np.imag(noise), '.')
plt.xlabel('Q')
plt.ylabel('I')
plt.show()