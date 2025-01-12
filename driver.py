import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import activation_functions
from neural_network import NeuralNetwork
from error_functions import cross_entropy_softmax
from activation_functions import sigmoid, relu,  identity
from mnist_loader import load_mnist
# Configurazioni iniziali
TRAIN_SIZE = 10000
VAL_SIZE = 5000
TEST_SIZE = 10000
MAX_EPOCHS = 100
HIDDEN_NODES = [16, 32, 64, 128, 256]
GL_ALPHA_VALUES = [0.01, 0.03, 0.05]
PQ_ALPHA_VALUES = [0.5, 1, 2]
PQ_K = 5
RPROP_ETA_PLUS = 1.1
RPROP_ETA_MINUS = 0.5
DELTA_MIN = 1e-16
DELTA_MAX = 50
ACTIVATIONS = [sigmoid, identity]

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(TRAIN_SIZE, VAL_SIZE, TEST_SIZE)

import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def run_experiment(nodes: int,
                   early_stopping: str | None,
                   alpha_gl: float = None,
                   alpha_pq: float = None):
    # Inizializziamo i valori per tenere traccia SOLO della miglior rete
    best_net_validation_error = float('inf')
    best_net_training_error = float('inf')
    best_net_error_training_history = []
    best_net_error_validation_history = []
    best_net_epoch = 0
    best_accuracy = 0.0

    # Definizione dei layer
    layers = [784, nodes, 10]

    # Assicuriamoci che esista la cartella "results"
    os.makedirs("results", exist_ok=True)

    # Eseguiamo più volte (ad es. 20) per selezionare la miglior rete
    for i in range(5):
        nn = NeuralNetwork(layers, ACTIVATIONS)
        error_training_history, error_validation_history, epoch_reached, min_err_val, min_err_train = nn.train(
            x_train, y_train, x_val, y_val,
            max_epochs=MAX_EPOCHS,
            error_function=cross_entropy_softmax,
            updating_rule="rprop",
            rprop_eta_minus=RPROP_ETA_MINUS,
            rprop_eta_plus=RPROP_ETA_PLUS,
            rprop_delta_min=DELTA_MIN,
            rprop_delta_max=DELTA_MAX,
            early_stopping_criteria=early_stopping,
            early_stopping_criteria_gl_alpha=alpha_gl,
            early_stopping_criteria_pq_alpha=alpha_pq,
            early_stopping_criteria_pq_k=PQ_K,
            log_progress=False
        )

        # Calcolo dell'accuracy sul test set
        accuracy = nn.compute_accuracy(x_test, y_test)

        # Controllo se questo run ha il minor errore di validazione
        if min_err_val < best_net_validation_error:
            best_net_validation_error = min_err_val
            best_net_training_error = min_err_train
            best_net_error_training_history = error_training_history[:]
            best_net_error_validation_history = error_validation_history[:]
            best_accuracy = accuracy
            best_net_epoch = epoch_reached

    # --------------------------------------------------
    #          Plot della migliore rete
    # --------------------------------------------------
    # Aggiungiamo alpha_gl e alpha_pq al titolo (se preferisci, puoi cambiare il formato)

    if early_stopping == "generalization_loss":
        title = f"Best Model - {nodes} nodes - EarlyStopping: Generalization Loss - Alpha: {alpha_gl}"
    elif early_stopping == "progress_quotient":
        title = f"Best Model - {nodes} nodes - EarlyStopping: Progress Quotient - Alpha: {alpha_pq}"
    else:
        title = f"Best Model - {nodes} nodes - No EarlyStopping"

    plot_title = title

    plt.figure(figsize=(10, 6))
    plt.plot(best_net_error_training_history, label='Errore Training')
    plt.plot(best_net_error_validation_history, label='Errore Validation')
    plt.xlabel('Epoche')
    plt.ylabel('Errore')
    plt.title(plot_title)
    plt.legend()
    plt.grid()

    # Annotazioni con i valori migliori
    annotation_text = (
        f"Accuracy: {best_accuracy:.4f}\n"
        f"Val Error: {best_net_validation_error:.4f}\n"
        f"Train Error: {best_net_training_error:.4f}\n"
        f"Epoch Reached: {best_net_epoch}"
    )
    plt.text(
        0.65, 0.65,  # Coordinata (x, y) in assi normalizzati [0, 1]
        annotation_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.6)
    )

    # --------------------------------------------------
    #     Salvataggio del grafico e del txt
    # --------------------------------------------------

    # Se preferisci, puoi evitare di scrivere "None" nei nomi e sostituirlo con "" o altro.
    alpha_gl_str = str(alpha_gl) if alpha_gl is not None else "None"
    alpha_pq_str = str(alpha_pq) if alpha_pq is not None else "None"

    # Nome del file includendo i parametri alpha

    filename_plot = (
        f"results/{nodes}_nodes_{early_stopping}_"
        f"gl_{alpha_gl_str}_pq_{alpha_pq_str}_earlyStopping.png"
    )
    plt.savefig(filename_plot, dpi=300)

    # File di testo con gli stessi parametri nel nome
    filename_txt = (
        f"results/{nodes}_nodes_{early_stopping}_"
        f"gl_{alpha_gl_str}_pq_{alpha_pq_str}_earlyStopping.txt"
    )
    with open(filename_txt, "w") as f:
        f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Best Validation Error: {best_net_validation_error:.4f}\n")
        f.write(f"Best Training Error: {best_net_training_error:.4f}\n")
        f.write(f"Epoch Reached: {best_net_epoch}\n")

    # Se desideri restituire i risultati in output
    return {
        "best_accuracy": best_accuracy,
        "best_validation_error": best_net_validation_error,
        "best_training_error": best_net_training_error,
        "best_epoch": best_net_epoch
    }







for nodes in HIDDEN_NODES:
    layers = [784, nodes, 10]
    print("Numero nodi: ",nodes, "Early stopping: None")
    run_experiment(nodes, None)
    for alpha_gl in GL_ALPHA_VALUES:
        print("Numero nodi: ",nodes,"Early stopping: generalization_loss Alpha: ",alpha_gl)
        run_experiment(nodes, "generalization_loss", alpha_gl)
    for alpha_pq in PQ_ALPHA_VALUES:
        print("Numero nodi: ",nodes,"Early stopping: progress_quotient Alpha: ",alpha_pq)
        run_experiment(nodes, "progress_quotient", alpha_pq=alpha_pq)



# Definizione della rete neurale

def run_experiment_0(nodes: int,
                   early_stopping: str| None,
                   alpha_gl: float = None,
                   alpha_pq: float = None):
    best_net_validation_error = float('inf')
    best_net_error_training_history = []
    best_net_error_validation_history = []
    epoch_reached_list = []
    error_training_list = []
    error_validation_list = []
    accuracy_list = []
    best_accuracy = 0.0
    best_net_epoch = 0
    layers = [784, nodes, 10]
    for i in range(2):
        nn = NeuralNetwork(layers, ACTIVATIONS)
        error_training_history, error_validation_history, epoch_reached, min_err_val, min_err_train = nn.train(
            x_train, y_train, x_val, y_val,
            max_epochs=MAX_EPOCHS,
            error_function=cross_entropy_softmax,
            updating_rule="rprop",
            rprop_eta_minus=RPROP_ETA_MINUS,
            rprop_eta_plus=RPROP_ETA_PLUS,
            rprop_delta_min=DELTA_MIN,
            rprop_delta_max=DELTA_MAX,
        )

        error_training_list.append(error_training_history[-1])
        error_validation_list.append(error_validation_history[-1])
        epoch_reached_list.append(epoch_reached)

        accuracy = nn.compute_accuracy(x_test, y_test)
        accuracy_list.append(accuracy)

        if min_err_val < best_net_validation_error:
            best_net_validation_error = min_err_val
            best_net_error_training_history = error_training_history[:]
            best_net_error_validation_history = error_validation_history[:]
            best_accuracy = accuracy
            best_net_epoch = epoch_reached

    mean_accuracy = np.mean(accuracy_list)
    mean_error_training = np.mean(error_training_list)
    mean_error_validation = np.mean(error_validation_list)

    # Plot degli errori
    plt.figure(figsize=(10, 6))
    plt.plot(error_training_history, label='Errore Training')
    plt.plot(error_validation_history, label='Errore Validation')
    plt.xlabel('Epoche')
    plt.ylabel('Errore')
    plt.title('Andamento dell\'errore durante il training miglior rete con ')
    plt.legend()
    plt.grid()
    plt.show()






