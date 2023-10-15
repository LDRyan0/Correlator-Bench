import sys
import os
import numpy as np
import matplotlib.pyplot as plt

results_dir = "results/"
img_dir = "img/"

# get every .csv given by the user

ants = []
chans = []
times = []
pols = []
input_sizes = []
output_sizes = []
in_reorders_ms = []
computes_ms = []
tri_reorders_ms = []
mwax_reorders_ms = []
totals_ms = []
compute_TOPS = []
total_TOPS = []
labels = []
errors_rms = []

files = [f for f in sys.argv if f[-4:] == ".csv"]

if not files:
    print("Please supply .csv files as arguments")
    exit(0)

# iterate over results from given correlators
for i, file in enumerate(files):
    # remove ".csv" for description
    description = file[:-4]

    # parse csv into 2D numpy array
    data = np.genfromtxt(os.path.join(results_dir, file), delimiter=',')

    # .csv layout:
    #       nstation, nfrequency, ntime, npol, input reorder (ms),compute (ms),tri reorder (ms), 
    #       mwax reorder (ms),total (ms),compute (TOPS),total (TOPS)
    ants.append(data[:,0])
    chans.append(data[:,1])
    times.append(data[:,2])
    pols.append(data[:,3])
    input_sizes.append(data[:,4])
    output_sizes.append(data[:,5])
    in_reorders_ms.append(data[:,6])
    computes_ms.append(data[:,7])
    tri_reorders_ms.append(data[:,8])
    mwax_reorders_ms.append(data[:,9])
    totals_ms.append(data[:,10])
    compute_TOPS.append(data[:,11])
    total_TOPS.append(data[:,12])
    errors_rms.append(data[:,13]) # will be 0 if user selected not to verify output
    labels.append(description)

fig, ax = plt.subplots()
for i, _ in enumerate(files):
    ax.plot(ants[i], compute_TOPS[i], label=labels[i])
ax.legend()
ax.set_xlabel('Number of antennas')
ax.set_ylabel('Performance (TOPS)')
plt.savefig(os.path.join(img_dir, "compute_performance.png"))

fig, ax = plt.subplots()
for i, _ in enumerate(files):
    ax.plot(ants[i], in_reorders_ms[i], label=labels[i])
ax.legend()
ax.set_xlabel('Number of antennas')
ax.set_ylabel('Input Reordering Time (ms)')
plt.savefig(os.path.join(img_dir, "input_reorder.png"))

fig, ax = plt.subplots()
for i, _ in enumerate(files):
    ax.plot(ants[i], computes_ms[i] + tri_reorders_ms[i], label=labels[i])
ax.legend()
ax.set_xlabel('Number of antennas')
ax.set_ylabel('Compute & Triangular Reorder Time (ms)')
plt.savefig(os.path.join(img_dir, "compute_tri.png"))

fig, ax = plt.subplots()
for i, _ in enumerate(files):
    ax.plot(output_sizes[i] // 1000000, errors_rms[i] / output_sizes[i], label=labels[i])
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('Output Size (millions)')
ax.set_ylabel('rms error per visibility')
plt.savefig(os.path.join(img_dir, "error.png"))