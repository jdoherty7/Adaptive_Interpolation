import matplotlib.pyplot as plt
import numpy as np

D = [4, 6, 9, 13]

Gnd = np.array([ 3.6 ,  3.11,  3.51, 0.93])
Mnd = np.array([10.81, 10.58, 10.52, 3.16])

Gcd = np.array([3.6, 3.47, 3.51, 1.74])
Mcd = np.array([7.84, 8.2, 7.96, 4.1])


plt.figure()
plt.title("Performance Total")
plt.subplot(211)
plt.ylabel("GFLOPS/s")
plt.plot(D, Gnd, c='b', label="store interval")
plt.plot(D, Gcd, c='r', label="calculate interval")
plt.subplot(212)
plt.ylabel("Memory Bandwidth (GB/s)")
plt.plot(D, Mnd, c='b', label="store interval")
plt.plot(D, Mcd, c='r', label="calculate interval")
plt.legend()
plt.show()


plt.figure()
plt.title("Performance Percent")
#plt.subplot(211)
#plt.ylabel("GFLOPS/s")
plt.ylabel("Percent of peak performance")
plt.plot(D, Gnd/35.2, c='b', label="store interval, GFLOPS")
plt.plot(D, Gcd/35.2, c='r', label="calculate interval, GFLOPS")
#plt.subplot(212)
#plt.ylabel("Memory Bandwidth (GB/s)")
plt.plot(D, Mnd/18.73, marker="o", c='b', label="store interval, MB")
plt.plot(D, Mcd/18.73, marker="o", c='r', label="calculate interval, MB")

#plt.plot(D, Gnd/35.2 +  Mnd/18.73, marker="s", c='b', label="total store interval")
#plt.plot(D, Gcd/35.2 +  Mcd/18.73, marker="s", c='r', label="total calculate interval")

plt.legend()

plt.show()