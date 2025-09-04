import numpy as np

freq_min = 125
freq_max = 2500
freq_num = 30

aA = 165.4
k = 0.88
a = 2.1

xmin = np.log10(freq_min / aA + k) / a

xmax = np.log10(freq_max / aA + k) / a

x_map = np.linspace(xmin, xmax, freq_num)
cfs = aA * (10**( a*x_map ) - k)

print(cfs)

            