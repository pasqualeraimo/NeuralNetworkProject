import numpy as np

def sum_of_squares(predictions: np.ndarray, targets: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola l'errore della somma dei quadrati (MSE) o la sua derivata rispetto alle predizioni.

    Parametri:
    predictions (np.ndarray): Array delle predizioni della rete neurale di dimensione (output_dim, batch_size).
    targets (np.ndarray): Array dei valori target di dimensione (output_dim, batch_size).
    derivative (bool): Se True, calcola la derivata dell'errore rispetto alle predizioni.

    Restituisce:
    np.ndarray:<
        - Se `derivative` è True, restituisce un array di dimensione (output_dim, batch_size) contenente la derivata dell'errore.
        - Altrimenti, restituisce un array scalare contenente l'errore complessivo (somma dei quadrati).
    """
    if derivative:
        return predictions - targets
    return 0.5 * np.sum((predictions - targets) ** 2)

def cross_entropy(predictions: np.ndarray, targets: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola l'errore di cross-entropy o la sua derivata rispetto alle predizioni.

    Parametri:
    predictions (np.ndarray): Array delle predizioni della rete neurale di dimensione (output_dim, batch_size).
    targets (np.ndarray): Array dei valori target di dimensione (output_dim, batch_size).
    derivative (bool): Se True, calcola la derivata dell'errore rispetto alle predizioni.

    Restituisce:
    np.ndarray:
        - Se `derivative` è True, restituisce un array di dimensione (output_dim, batch_size) contenente la derivata dell'errore.
        - Altrimenti, restituisce un array scalare contenente l'errore complessivo (cross-entropy).
    """
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    if derivative:
        return - targets / predictions
    return -np.sum(targets * np.log(predictions))

def cross_entropy_softmax(predictions: np.ndarray, targets: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calcola la cross-entropy con softmax o la sua derivata rispetto alle predizioni.

    Parametri:
    predictions (np.ndarray): Array delle predizioni della rete neurale di dimensione (output_dim, batch_size).
    targets (np.ndarray): Array dei valori target di dimensione (output_dim, batch_size).
    derivative (bool): Se True, calcola la derivata dell'errore rispetto alle predizioni.

    Restituisce:
    np.ndarray:
        - Se `derivative` è True, restituisce un array di dimensione (output_dim, batch_size) contenente il gradiente combinato di softmax e cross-entropy.
        - Altrimenti, restituisce un array scalare che rappresenta l'errore complessivo.
    """
    epsilon = 1e-12
    softmax_outputs = softmax(predictions)
    softmax_outputs = np.clip(softmax_outputs, epsilon, 1)
    if derivative:
        return softmax_outputs - targets
    return -np.sum(targets * np.log(softmax_outputs))

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Calcola la funzione softmax.

    Parametri:
    x (np.ndarray): Array di input, di dimensione arbitraria.

    Restituisce:
    np.ndarray: Output della funzione softmax, stessa dimensione di x.
    """
    exp_vals = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)