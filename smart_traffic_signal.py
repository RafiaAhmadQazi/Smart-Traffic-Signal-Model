"""
Smart Traffic Signal using a Single-Layer Perceptron
Educational demonstration only — not for controlling real traffic lights.
"""

import numpy as np

# 1. TRAINING DATA
# x1 = long vehicle queue? (1 = Yes, 0 = No)
# x2 = long waiting time? (1 = Yes, 0 = No)
# x3 = heavy traffic flow? (1 = Yes, 0 = No)
# y  = extra green time recommended? (1 = Yes, 0 = No)
#
# These are illustrative classroom examples, not measured real traffic data.

X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# 2. PERCEPTRON MODEL
class Perceptron:
    def __init__(self, number_of_inputs, learning_rate=1, epochs=20):
        self.weights = np.zeros(number_of_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    # TLU-style step activation function
    def activation(self, weighted_sum):
        if weighted_sum >= 0:
            return 1
        return 0

    # Forward prediction for one example
    def predict_one(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.activation(weighted_sum)

    # Learn from the training examples
    def fit(self, X, y):
        for epoch in range(self.epochs):
            errors = 0

            for inputs, target in zip(X, y):
                prediction = self.predict_one(inputs)
                error = target - prediction

                # Perceptron learning rule
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

                if error != 0:
                    errors += 1

            print(f"Epoch {epoch + 1}: {errors} errors")

            if errors == 0:
                break


# 3. CREATE AND TRAIN THE MODEL
model = Perceptron(number_of_inputs=3)
model.fit(X, y)

print("\nLearned weights:", model.weights)
print("Learned bias:", model.bias)


# 4. TEST WITH EXAMPLE TRAFFIC CONDITIONS
test_cases = np.array([
    [0, 0, 0],  # Short queue, short wait, light traffic
    [1, 0, 0],  # Long queue only
    [0, 1, 1],  # Long wait and heavy traffic
    [1, 1, 1]   # All three conditions present
])

print("\nSMART TRAFFIC SIGNAL RESULTS")

for traffic in test_cases:
    prediction = model.predict_one(traffic)
    print("Input:", traffic, end=" -> ")

    if prediction == 1:
        print("Extra green time recommended")
    else:
        print("No extra green time recommended")
