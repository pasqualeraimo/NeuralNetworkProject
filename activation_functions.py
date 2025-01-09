import numpy as np

def sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola la funzione sigmoid o la sua derivata.

    La funzione sigmoid limita i valori di input in un intervallo compreso tra 0 e 1.

    Parametri:
    x (np.ndarray): Array di input di dimensione arbitraria.
    derivative (bool): Se True, calcola la derivata della funzione sigmoid.

    Restituisce:
    np.ndarray: Array di output della stessa dimensione di x, contenente:
        - I valori della funzione sigmoid applicata all'input, oppure
        - La derivata della funzione sigmoid.
    """
    sigmoid_output = 1 / (1 + np.exp(-x))
    if derivative:
        return sigmoid_output * (1 - sigmoid_output)
    return sigmoid_output

def identity(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola la funzione identità o la sua derivata.

    La funzione identità restituisce l'input senza modifiche.

    Parametri:
    x (np.ndarray): Array di input di dimensione arbitraria.
    derivative (bool): Se True, calcola la derivata della funzione identità.

    Restituisce:
    np.ndarray: Array di output della stessa dimensione di x, contenente:
        - I valori della funzione identità applicata all'input, oppure
        - La derivata della funzione identità.
    """
    if derivative:
        return np.ones_like(x)
    return x

def relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola la funzione ReLU (Rectified Linear Unit) o la sua derivata.

    La funzione ReLU restituisce 0 per valori negativi e l'input originale per
    valori positivi.

    Parametri:
    x (np.ndarray): Array di input di dimensione arbitraria.
    derivative (bool): Se True, calcola la derivata della funzione ReLU.

    Restituisce:
    np.ndarray: Array di output della stessa dimensione di x, contenente:
        - I valori della funzione ReLU applicata all'input, oppure
        - La derivata della funzione ReLU.
    """
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def tanh(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola la funzione tanh (tangente iperbolica) o la sua derivata.

    La funzione tanh normalizza i valori di input in un intervallo compreso tra
    -1 e 1.

    Parametri:
    x (np.ndarray): Array di input di dimensione arbitraria.
    derivative (bool): Se True, calcola la derivata della funzione tanh.

    Restituisce:
    np.ndarray: Array di output della stessa dimensione di x, contenente:
        - I valori della funzione tanh applicata all'input, oppure
        - La derivata della funzione tanh.
    """
    tanh_output = np.tanh(x)
    if derivative:
        return 1 - tanh_output**2
    return tanh_output