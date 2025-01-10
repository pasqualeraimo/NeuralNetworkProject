import numpy as np
from network_types import ActivationFunction, ErrorFunction

class NeuralNetwork:
    def __init__(self,
                 layers: list[int],
                 activation_functions: list[ActivationFunction],
                 mean: float = 0.0,
                 standard_deviation: float = 0.1):
        """
        Inizializza la struttura della rete neurale.

        Parametri:
        layers (list[int]): Numero di nodi in ogni strato della rete.
        activation_functions (list[ActivationFunction]): Funzioni di attivazione per ogni strato.
        mean (float): Media della distribuzione normale utilizzata per inizializzare i parametri. Default 0.0.
        standard_deviation (float): Deviazione standard della distribuzione normale per l'inizializzazione. Default 0.1.

        Note:
        I pesi e i bias della rete sono inizializzati usando una distribuzione normale con i parametri configurabili
        `mean` e `standard_deviation`.
        """
        self.layers = layers
        self.activation_functions = activation_functions
        self.weights, self.biases = self._init_parameters_normal_distribution(mean, standard_deviation)

    def _init_parameters_normal_distribution(self,
                                             mean: float = 0.0,
                                             standard_deviation: float = 0.1) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Inizializza i pesi e i bias con una distribuzione normale.

        Parametri:
        mean (float): Media della distribuzione normale. Default 0.0.
        standard_deviation (float): Deviazione standard della distribuzione normale. Default 0.1.

        Restituisce:
        tuple[list[np.ndarray], list[np.ndarray]]: Liste dei pesi e dei bias inizializzati.
            - weights: Liste di matrici dei pesi, ogni matrice ha dimensione (nodi_strato_successivo, nodi_strato_corrente).
            - biases: Liste di vettori dei bias, ogni vettore ha dimensione (nodi_strato_successivo, 1).
        """
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for i in range(len(self.layers) - 1):
            # Inizializza i pesi con una matrice di dimensione (nodi_strato_successivo, nodi_strato_corrente)
            weights.append(np.random.normal(loc=mean, scale=standard_deviation, size=(self.layers[i + 1], self.layers[i])))
            # Inizializza i bias con una matrice colonna di dimensione (nodi_strato_successivo, 1)
            biases.append(np.random.normal(loc=mean, scale=standard_deviation, size=(self.layers[i + 1], 1)))
        return weights, biases

    def forward_propagation(self,
                            input_samples: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Esegue la forward-propagation.

        Parametri:
        input_samples (np.ndarray): Array di input alla rete neurale, di dimensione (nodi_input, batch_size).

        Restituisce:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - activations: Lista delle attivazioni per ogni strato (incluso l'input iniziale).
              Ogni matrice di attivazione ha dimensione (nodi_strato, batch_size).
              La lista ha dimensione (numero_layer).
            - pre_activations: Lista delle pre-attivazioni per ogni strato (escluso l'input layer).
              Ogni matrice di pre-attivazione ha dimensione (nodi_strato_successivo, batch_size).
              La lista ha dimensione (numero_layer - 1).

        Note:
        Questa funzione assume che le dimensioni degli input siano compatibili con il primo strato della rete neurale.
        """
        activations = [input_samples]
        pre_activations = []
        for i in range(len(self.layers) - 1):
            # Calcola la matrice delle pre-attivazioni z = W * a + b
            # z ha dimensione (nodi_strato_successivo, batch_size)
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            pre_activations.append(z)
            # Applica la funzione di attivazione
            # a ha dimensione (nodi_strato_successivo, batch_size)
            a = self.activation_functions[i](z, False)
            activations.append(a)
        return activations, pre_activations

    def backward_propagation(self,
                             targets: np.ndarray,
                             error_function: ErrorFunction,
                             activations: list[np.ndarray],
                             pre_activations: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Esegue la back-propagation per calcolare i gradienti dei pesi e dei bias.

        Parametri:
        activations (list[np.ndarray]): Lista di matrici delle attivazioni per ogni strato.
        pre_activations (list[np.ndarray]): Lista di matrici delle pre-attivazioni per ogni strato.
        targets (np.ndarray): Valori target per il calcolo dell'errore, di dimensione (nodi_output, batch_size).
        error_function (ErrorFunction): Funzione di errore utilizzata per calcolare l'errore.

        Restituisce:
        tuple[list[np.ndarray], list[np.ndarray]]:
            - weight_gradients: Gradienti dei pesi per ogni strato, ogni matrice ha dimensione (nodi_strato_successivo, nodi_strato_corrente).
            - bias_gradients: Gradienti dei bias per ogni strato, ogni vettore ha dimensione (nodi_strato_successivo, 1).
        """

        delta = []
        weight_gradients: list[np.ndarray] = []
        bias_gradients: list[np.ndarray] = []

        # Calcolo dei delta (errore locale) per ogni strato
        for i in range(len(self.layers) - 1):
            if i == 0:
                # Calcolo del delta per l'ultimo strato
                # delta ha dimensione (nodi_output, batch_size)
                delta.append(self.activation_functions[-1](pre_activations[-1], True) *
                             error_function(activations[-1], targets, True))
            else:
                # Calcolo del delta per gli strati nascosti
                # delta ha dimensione (nodi_strato_corrente, batch_size)
                delta.append(self.activation_functions[-i - 1](pre_activations[-i - 1], True) *
                             np.dot(self.weights[-i].T, delta[i - 1]))

        # Inversione dei delta per allinearli agli strati
        delta.reverse()

        # Calcolo dei gradienti dei pesi e dei bias
        for i in range(len(self.layers) - 1):
            # Gradiente dei pesi: dimensione (nodi_strato_successivo, nodi_strato_corrente)
            weight_gradients.append(np.dot(delta[i], activations[i].T))
            # Gradiente dei bias: dimensione (nodi_strato_successivo, 1)
            bias_gradients.append(np.sum(delta[i], axis=1, keepdims=True))

        return weight_gradients, bias_gradients