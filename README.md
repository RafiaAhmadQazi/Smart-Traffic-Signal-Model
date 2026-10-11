# Smart Traffic Signal using a Single-Layer Perceptron

## Project idea
This educational project demonstrates how a single-layer Perceptron can classify simplified traffic conditions and recommend whether extra green time is needed.

## Inputs
- `x1`: Is the vehicle queue long? (`1` = Yes, `0` = No)
- `x2`: Is waiting time high? (`1` = Yes, `0` = No)
- `x3`: Is traffic flow heavy? (`1` = Yes, `0` = No)

## Output
- `0`: No extra green time recommended
- `1`: Extra green time recommended

## Important note about the data
The dataset is a small, illustrative classroom dataset. The labels are examples created for learning and are not collected from real roads. This program is not validated for real traffic management and must not be used to control actual traffic lights.

## Model concepts demonstrated
1. Training inputs and target labels
2. Weights and bias
3. Weighted sum: `z = np.dot(inputs, weights) + bias`
4. TLU-style step activation function
5. Forward prediction
6. Perceptron learning rule
7. Epochs and error counting
8. Testing the trained model on example inputs

## Requirements
- Python 3
- NumPy

Install NumPy:

```bash
python -m pip install numpy
```

## Run the project

```bash
python smart_traffic_signal.py
```

## GitHub upload
After creating an empty repository on GitHub, run these commands in the project folder:

```bash
git init
git add smart_traffic_signal.py README.md .gitignore
git commit -m "feat: add smart traffic signal perceptron"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your own GitHub repository.
