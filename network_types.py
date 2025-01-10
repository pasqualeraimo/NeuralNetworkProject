import numpy as np
from typing import Callable

# Tipo per la funzione di attivazione, che accetta:
# - Un array numpy.ndarray, che rappresenta l'input alla funzione.
# - Un booleano, che indica se si desidera calcolare la derivata della funzione.
# Restituisce un array numpy.ndarray che rappresenta l'output elaborato dalla funzione di attivazione o la sua derivata.
ActivationFunction = Callable[[np.ndarray, bool], np.ndarray]

# Tipo per la funzione di errore, che accetta:
# - Un array numpy.ndarray, che rappresenta la predizione dalla rete neurale.
# - Un array numpy.ndarray, che rappresenta i valori attesi.
# - Un booleano, che indica se si desidera calcolare la derivata della funzione di errore.
# Restituisce un array numpy.ndarray che rappresenta l'errore calcolato o la sua derivata.
ErrorFunction = Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
