import os
numcpus = 4
max_freq = "2900000"
max_freq = "2800000"
#max_freq = "2200000"

for i in range(numcpus):
    dir = "/sys/devices/system/cpu/cpu"+str(int(i))+"/cpufreq/scaling_max_freq"
    os.system("echo "+max_freq+" > "+dir)