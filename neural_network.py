from typing import Literal

import numpy as np
from network_types import ActivationFunction, ErrorFunction


class NeuralNetwork:
    def __init__(self,
                 layers: list[int],
                 activation_functions: list[ActivationFunction],
                 mean: float = 0.0,
                 standard_deviation: float = 0.1,
                 init_parameters_seed: int = None) -> None:
        """
        Inizializza la struttura della rete neurale.

        Parametri:
        layers (list[int]): Numero di nodi in ogni strato della rete.
        activation_functions (list[ActivationFunction]): Funzioni di attivazione per ogni strato.
        mean (float): Media della distribuzione normale utilizzata per inizializzare i parametri. Default 0.0.
        standard_deviation (float): Deviazione standard della distribuzione normale per l'inizializzazione. Default 0.1.
        init_parameters_seed (int): Seed per il generatore di numeri casuali per l'inizializzazione dei pesi e dei bias. Default None.

        Note:
        I pesi e i bias della rete sono inizializzati usando una distribuzione normale con i parametri configurabili
        `mean` e `standard_deviation`.
        """
        self.layers = layers
        self.activation_functions = activation_functions
        self.weights, self.biases = self._init_parameters_normal_distribution( init_parameters_seed, mean, standard_deviation)

    def _init_parameters_normal_distribution(self,
                                             seed: int = None,
                                             mean: float = 0.0,
                                             standard_deviation: float = 0.1) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Inizializza i pesi e i bias con una distribuzione normale.

        Parametri:
        mean (float): Media della distribuzione normale. Default 0.0.
        standard_deviation (float): Deviazione standard della distribuzione normale. Default 0.1.
        seed (int): Seed per il generatore di numeri casuali. Default None.

        Restituisce:
        tuple[list[np.ndarray], list[np.ndarray]]: Liste dei pesi e dei bias inizializzati.
            - weights: Liste di matrici dei pesi, ogni matrice ha dimensione (nodi_strato_successivo, nodi_strato_corrente).
            - biases: Liste di vettori dei bias, ogni vettore ha dimensione (nodi_strato_successivo, 1).
        """
        if seed is not None:
            np.random.seed(seed)  # Imposta il seed per la generazione casuale

        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for i in range(len(self.layers) - 1):
            # Inizializza i pesi con una matrice di dimensione (nodi_strato_successivo, nodi_strato_corrente)
            weights.append(
                np.random.normal(loc=mean, scale=standard_deviation, size=(self.layers[i + 1], self.layers[i])))
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
        activations (list[np.ndarray]): Lista di matrici delle attivazioni per ogni strato. La lista ha dimensione (numero_layer).
        pre_activations (list[np.ndarray]): Lista di matrici delle pre-attivazioni per ogni strato. La lista ha dimensione (numero_layer - 1).
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

    def update_parameters_stochastic_gradient_descent(self,
                                                      weight_gradients: list[np.ndarray],
                                                      bias_gradients: list[np.ndarray],
                                                      learning_rate: float = 0.001):
        """
        Aggiorna i pesi e i bias utilizzando l'updating rule 'stochastic gradient descent'.

        Parametri:
        weight_gradients (list[np.ndarray]): Gradienti dei pesi, ogni matrice ha dimensione (nodi_strato_successivo, nodi_strato_corrente).
        bias_gradients (list[np.ndarray]): Gradienti dei bias, ogni vettore ha dimensione (nodi_strato_successivo, 1).
        learning_rate (float): Tasso di apprendimento per l'aggiornamento dei parametri. Default 0.001.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]

    def update_parameters_rprop(self,
                                weight_gradients: list[np.ndarray],
                                bias_gradients: list[np.ndarray],
                                prev_weight_gradients: list[np.ndarray],
                                prev_bias_gradients: list[np.ndarray],
                                weight_delta: list[np.ndarray],
                                bias_delta: list[np.ndarray],
                                eta_minus: float = 0.5,
                                eta_plus: float = 1.1,
                                delta_min: float = 1e-06,
                                delta_max: float = 50):
        """
        Aggiorna i parametri della rete neurale utilizzando l'algoritmo RProp.

        Parametri:
        weight_gradients (list[np.ndarray]): Gradienti correnti dei pesi.
        bias_gradients (list[np.ndarray]): Gradienti correnti dei bias.
        prev_weight_gradients (list[np.ndarray]): Gradienti dei pesi dalla precedente iterazione.
        prev_bias_gradients (list[np.ndarray]): Gradienti dei bias dalla precedente iterazione.
        weight_delta (list[np.ndarray]): Passi adattivi correnti per i pesi.
        bias_delta (list[np.ndarray]): Passi adattivi correnti per i bias.
        eta_plus (float): Fattore di incremento per il passo adattivo. Default 1.1.
        eta_minus (float): Fattore di decremento per il passo adattivo. Default 0.5.
        delta_min (float): Valore minimo consentito per il passo adattivo. Default 1e-06.
        delta_max (float): Valore massimo consentito per il passo adattivo. Default 50.

        """
        for i in range(len(self.layers)-1):

            if prev_weight_gradients:  # controlla la lista dei gradienti non sia vuota (ovvero c'è stata almeno un'altra epoca)
                sign_change_weights = weight_gradients[i] * prev_weight_gradients[i]

                weight_delta[i] = np.where(
                    sign_change_weights > 0,
                    np.minimum(weight_delta[i] * eta_plus, delta_max),  # Incrementa delta se segno è lo stesso
                    np.where(
                        sign_change_weights < 0,
                        np.maximum(weight_delta[i] * eta_minus, delta_min),  # Riduci delta se segno cambia
                        weight_delta[i]  # Mantieni delta invariato se non c'è cambiamento
                    )
                )

                sign_change_bias = bias_gradients[i] * prev_bias_gradients[i]
                bias_delta[i] = np.where(
                    sign_change_bias > 0,
                    np.minimum(bias_delta[i] * eta_plus, delta_max),
                    np.where(
                        sign_change_bias < 0,
                        np.maximum(bias_delta[i] * eta_minus, delta_min),
                        bias_delta[i]))

            self.weights[i] -= np.sign(weight_gradients[i]) * weight_delta[i]
            self.biases[i] -= np.sign(bias_gradients[i]) * bias_delta[i]

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_validation: np.ndarray,
              y_validation: np.ndarray,
              max_epochs: int,
              error_function: ErrorFunction,
              updating_rule: Literal["sgd", "rprop"] = "sgd",
              sgd_learning_rate: float = 0.001,
              rprop_eta_minus: float = 0.5,
              rprop_eta_plus: float = 1.1,
              rprop_delta_min: float = 1e-06,
              rprop_delta_max: float = 50,
              early_stopping_criteria: Literal["generalization_loss", "prediction_quality"] | None = None,
              early_stopping_criteria_gl_alpha: float = 0.01,
              early_stopping_criteria_pq_k: int = 5,
              log_progress: bool = True) -> tuple[list[float], list[float]]:
        """
        Allena la rete neurale utilizzando il training set e valuta il modello sul validation set.

        Parametri:
        x_train (np.ndarray): Input del training set, dimensione (nodi_input, numero_campioni_train).
        y_train (np.ndarray): Target del training set, dimensione (nodi_output, numero_campioni_train).
        x_validation (np.ndarray): Input del validation set, dimensione (nodi_input, numero_campioni_validation).
        y_validation (np.ndarray): Target del validation set, dimensione (nodi_output, numero_campioni_validation).
        max_epochs (int): Numero massimo di epoche di allenamento.
        error_function (ErrorFunction): Funzione di errore per calcolare la perdita.
        updating_rule (Literal["sgd", "rprop"]): Regola di aggiornamento dei pesi ("sgd" o "rprop"). Default "sgd".
        sgd_learning_rate (float): Tasso di apprendimento per SGD. Ignorato se si utilizza RProp. Default 0.001.
        rprop_eta_minus (float): Fattore di decremento per RProp. Default 0.5.
        rprop_eta_plus (float): Fattore di incremento per RProp. Default 1.1.
        rprop_delta_min (float): Valore minimo del delta per RProp. Default 1e-06.
        rprop_delta_max (float): Valore massimo del delta per RProp. Default 50.
        early_stopping_criteria (Literal["generalization_loss", "prediction_quality"] | None): Criterio per l'early stopping.
        early_stopping_criteria_gl_alpha (float): Soglia per la perdita di generalizzazione (alpha) nel criterio "generalization_loss". Default 0.01.
        early_stopping_criteria_pq_k (int): Numero di epoche da considerare per il criterio "prediction_quality". Default 5.
        log_progress (bool): Se True, stampa a video il progresso del training. Default True.

        Restituisce:
        tuple[list[float], list[float]]:
            - error_training_history: Cronologia degli errori sul training set dopo ogni epoca.
            - error_validation_history: Cronologia degli errori sul validation set dopo ogni epoca.

        Note:
        - Se si utilizza RProp, i parametri relativi a SGD (come il learning rate) vengono ignorati.
        - L'early stopping interrompe l'allenamento anticipatamente se viene soddisfatto il criterio specificato.
        """
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]

        error_training_history = []
        error_validation_history = []

        prev_weight_gradients = []
        prev_bias_gradients = []
        weight_delta = [np.full_like(w, 0.01) for w in self.weights]
        bias_delta = [np.full_like(b, 0.01) for b in self.biases]

        min_error_validation = float('inf')

        for epoch in range(max_epochs):
            activations_train, pre_activations_train = self.forward_propagation(x_train)

            weight_gradients, bias_gradients = self.backward_propagation(
                targets=y_train,
                error_function=error_function,
                activations=activations_train,
                pre_activations=pre_activations_train
            )

            if updating_rule == "sgd":
                self.update_parameters_stochastic_gradient_descent(weight_gradients, bias_gradients,
                                                                   sgd_learning_rate)
            elif updating_rule == "rprop":
                self.update_parameters_rprop(weight_gradients,
                                             bias_gradients,
                                             prev_weight_gradients,
                                             prev_bias_gradients,
                                             weight_delta,
                                             bias_delta,
                                             rprop_eta_minus,
                                             rprop_eta_plus,
                                             rprop_delta_min,
                                             rprop_delta_max)
                prev_weight_gradients = weight_gradients
                prev_bias_gradients = bias_gradients

            # Calcolo errore su training e validation, dopo l'aggiornamento

            # Training
            activations_train, _ = self.forward_propagation(x_train)
            error_training = error_function(activations_train[-1], y_train, False)
            error_training_history.append(error_training)

            # Validation
            activations_validation, _ = self.forward_propagation(x_validation)
            error_validation = error_function(activations_validation[-1], y_validation, False)
            error_validation_history.append(error_validation)

            # Stampa a video (opzionale)
            if log_progress:
                print(
                    f"Epoch {epoch + 1}/{max_epochs}: Training error = {error_training:.6f}, Validation error = {error_validation:.6f}")

            # Salvataggio dei pesi e bias se migliora il validation error
            if error_validation < min_error_validation:
                min_error_validation = error_validation
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]

            generalization_loss = 100 * ((error_validation / min_error_validation) - 1)
            if early_stopping_criteria == "generalization_loss":
                if generalization_loss > early_stopping_criteria_gl_alpha:
                    print("Early stopping criteria reached")
                    break
            elif early_stopping_criteria == "prediction_quality":
                if len(error_training_history) >= early_stopping_criteria_pq_k:
                    pq_min_error_training = error_training_history[-1]
                    sum = 0
                    for i in range(early_stopping_criteria_pq_k):
                        if error_training_history[-i - 1] < pq_min_error_training:
                            pq_min_error_training = error_training_history[-i - 1]
                        sum += error_training_history[-i - 1]
                    pq_progress = 1000 * ((sum / (early_stopping_criteria_pq_k * pq_min_error_training)) - 1)
                    if (generalization_loss / pq_progress) > early_stopping_criteria_gl_alpha:
                        print("Early stopping criteria reached")
                        break

        # Ripristino dei migliori pesi e bias (early stopping)
        self.weights = best_weights
        self.biases = best_biases

        return error_training_history, error_validation_history

    def compute_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Valuta l'accuracy della rete neurale su un test set.

        Parametri:
        x_test (np.ndarray): Input del test set, dimensione (nodi_input, numero_campioni_test).
        y_test (np.ndarray): Target del test set, dimensione (nodi_output, numero_campioni_test).

        Restituisce:
        float: Accuracy calcolata sul test set.
        """

        activations_test, _ = self.forward_propagation(x_test)

        # Predizioni: scegli l'indice del massimo valore (classe con maggiore probabilità)
        predictions = np.argmax(activations_test[-1], axis=0)

        # Target: scegli l'indice del massimo valore nel target (one-hot encoding)
        true_labels = np.argmax(y_test, axis=0)

        # Calcola l'accuracy
        accuracy = np.mean(predictions == true_labels)

        return accuracy