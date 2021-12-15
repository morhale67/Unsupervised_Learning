from sklearn.datasets import load_digits

def load_data():
    """importing the dataset"""
    digits = load_digits()
    x_data = digits.data
    y_data = digits.target
    return x_data, y_data
