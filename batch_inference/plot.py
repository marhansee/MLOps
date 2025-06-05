import matplotlib.pyplot as plt

batch_sizes = [1, 2, 4, 8, 16, 32]
latencies = [83.89, 170.15, 364.18, 759.81, 1492.79, 2992.73]  # ms per batch
throughputs = [11.92, 11.75, 10.98, 10.53, 10.34, 10.31]  # images/sec

plt.figure(figsize=(10,6))
plt.plot(latencies, throughputs, marker='o', color='blue', linestyle='-')

for i, batch in enumerate(batch_sizes):
    plt.annotate(str(batch), (latencies[i], throughputs[i]),
                 textcoords="offset points", xytext=(5,5), ha='left', fontsize=15, color='red')

plt.title("Latency vs Throughput for Different Batch Sizes", fontsize=20)
plt.xlabel("Latency per Batch (ms)", fontsize=20)
plt.ylabel("Throughput (images/sec)", fontsize=20)
plt.grid(True)
plt.show()