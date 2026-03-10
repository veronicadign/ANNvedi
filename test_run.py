from linear_ann.wrapper import LinearANN
import numpy as np

print("Starting test...")

data = np.random.rand(100, 32).astype(np.float32)

index = LinearANN()
index.fit(data)

q = data[0]

print("Nearest:", index.query(q, 5))
print("Distance computations:", index.total_distances_count())

print("Test finished")