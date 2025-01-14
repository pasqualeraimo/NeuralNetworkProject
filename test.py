from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import os

from numpy import floating

from neural_network import NeuralNetwork
from error_functions import cross_entropy_softmax
from activation_functions import sigmoid, relu,  identity
from mnist_loader import load_mnist

TRAIN_SIZE = 10000
VAL_SIZE = 5000
TEST_SIZE = 10000
MAX_EPOCHS = 100
HIDDEN_NODES = [16, 32, 64, 128, 256]

GL_ALPHA_VALUES = [0.01, 0.03, 0.05, 0.08]

PQ_ALPHA_VALUES = [0.1, 0.5, 1, 1.5]

PQ_K = 5
RPROP_ETA_PLUS = 1.1
RPROP_ETA_MINUS = 0.5
DELTA_MIN = 1e-16
DELTA_MAX = 50
ACTIVATIONS = [sigmoid, identity]
NUM_REPEATS = 10

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist(TRAIN_SIZE, VAL_SIZE, TEST_SIZE)

os.makedirs("results", exist_ok=True)

def run_experiment(nodes: int,
                   early_stopping: str | None,
                   alpha_gl: float = None,
                   alpha_pq: float = None,
                   num_repeats: int = 5)-> tuple[Any, Any, Any, Any, floating[Any], floating[Any], int, floating[Any]]:
    layers = [784, nodes, 10]

    accuracy_list = []
    epoch_reached_list = []
    error_training_matrix = np.full((num_repeats, MAX_EPOCHS), np.nan)
    error_validation_matrix = np.full((num_repeats, MAX_EPOCHS), np.nan)

    for i in range(num_repeats):
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

        accuracy = nn.compute_accuracy(x_test, y_test)
        accuracy_list.append(accuracy)
        epoch_reached_list.append(epoch_reached)
        error_training_matrix[i, :len(error_training_history)] = error_training_history
        error_validation_matrix[i, :len(error_validation_history)] = error_validation_history

    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    mean_epoch = int(np.floor(np.mean(epoch_reached_list)))
    std_epoch = np.std(epoch_reached_list)
    mean_error_training = np.nanmean(error_training_matrix, axis = 0)
    mean_error_training = mean_error_training[:mean_epoch]
    std_error_training = np.nanstd(error_training_matrix, axis = 0)
    std_error_training = std_error_training[:mean_epoch]
    mean_error_validation = np.nanmean(error_validation_matrix, axis = 0)
    mean_error_validation = mean_error_validation[:mean_epoch]
    std_error_validation = np.nanstd(error_validation_matrix, axis = 0)
    std_error_validation = std_error_validation[:mean_epoch]

    best_epoch_training = np.argmin(mean_error_training)
    best_epoch_validation = np.argmin(mean_error_validation)

    # Creazione e salvataggio plot

    if early_stopping == "generalization_loss":
        plot_title = f"Average Model - {nodes} nodes - EarlyStopping: Generalization Loss - Alpha: {alpha_gl}"
    elif early_stopping == "progress_quotient":
        plot_title = f"Average Model - {nodes} nodes - EarlyStopping: Progress Quotient - Alpha: {alpha_pq}"
    else:
        plot_title = f"Average Model - {nodes} nodes - No EarlyStopping"

    alpha_gl_str = str(alpha_gl) if alpha_gl is not None else "None"
    alpha_pq_str = str(alpha_pq) if alpha_pq is not None else "None"

    filename_plot = (
        f"results/{nodes}_nodes_{early_stopping}_"
        f"gl_{alpha_gl_str}_pq_{alpha_pq_str}_earlyStopping.png"
    )

    save_plot_error_history(mean_error_training,
                            std_error_training,
                            mean_error_validation,
                            std_error_validation,
                            mean_epoch,
                            std_epoch,
                            mean_accuracy,
                            std_accuracy,
                            plot_title,
                            filename_plot,
                            best_epoch_training,
                            best_epoch_validation)

    # Creazione e salvataggio txt

    filename_txt = (
        f"results/{nodes}_nodes_{early_stopping}_"
        f"gl_{alpha_gl_str}_pq_{alpha_pq_str}_earlyStopping.txt"
    )
    difference = mean_epoch - best_epoch_validation
    with open(filename_txt, "w") as f:
        f.write(f"Mean Accuracy: {mean_accuracy:.4f} ±{std_accuracy:.2f}\n")
        f.write(f"Best Validation Error: {np.min(mean_error_validation):.4f}\n")
        f.write(f"Best Training Error: {np.min(mean_error_training):.4f}\n")
        f.write(f"Epoch Reached: {mean_epoch} ±{std_epoch:.2f}\n")
        f.write(f"Best Epoch Validation: {best_epoch_validation}\n")
        f.write(f"Best Epoch Training: {best_epoch_training}\n")
        f.write(f"Difference: {difference}\n")





    return mean_error_training, std_error_training, mean_error_validation, std_error_validation, mean_accuracy, std_accuracy, mean_epoch, std_epoch


def save_plot_error_history(
        error_training,
        std_error_training,
        error_validation,
        std_error_validation,
        epoch_reached,
        std_epoch_reached,
        accuracy,
        std_accuracy,
        plot_title: str,
        filename_plot: str,
        best_epoch_training: int,
        best_epoch_validation: int):

    epochs_axis = np.arange(1, len(error_training) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot( epochs_axis,error_training, label='Training Error', color='blue')
    plt.fill_between(epochs_axis, error_training - std_error_training, error_training + std_error_training, alpha=0.2, color='blue')
    plt.plot(epochs_axis, error_validation, label='Validation Error', color='orange')
    plt.fill_between(epochs_axis, error_validation - std_error_validation, error_validation + std_error_validation, alpha=0.2, color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title(plot_title)
    plt.legend()
    plt.grid()

    plt.scatter(best_epoch_training, error_training[best_epoch_training - 1], color='blue', label='Best Train Error',
                zorder=5)
    plt.scatter(best_epoch_validation, error_validation[best_epoch_validation - 1], color='orange',
                label='Best Val Error', zorder=5)

    plt.annotate(
        f"Epoch {best_epoch_training}",
        (best_epoch_training, error_training[best_epoch_training - 1]),
        textcoords="offset points", xytext=(-30, 10), ha='center', color='blue'
    )
    plt.annotate(
        f"Epoch {best_epoch_validation}",
        (best_epoch_validation, error_validation[best_epoch_validation - 1]),
        textcoords="offset points", xytext=(30, -20), ha='center', color='orange'
    )
    plt.legend()

    annotation_text = (
        f"Accuracy: {accuracy:.4f} ±{std_accuracy:.2f} \n"
        f"Best Val Error: {np.min(error_validation):.4f}\n"
        f"Best Train Error: {np.min(error_training):.4f}\n"
        f"Epoch Reached: {epoch_reached} ±{std_epoch_reached:.2f}"
    )

    plt.text(
        0.5, 0.83,  # Coordinata (x, y) in assi normalizzati [0, 1]
        annotation_text,
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.6)
    )

    plt.savefig(filename_plot, dpi=300)

def run():
    for nodes in HIDDEN_NODES:
        print("Numero nodi: ", nodes, "Early stopping: None")
        run_experiment(nodes, None, num_repeats=NUM_REPEATS)
        for alpha_gl in GL_ALPHA_VALUES:
            print("Numero nodi: ", nodes, "Early stopping: generalization_loss Alpha: ", alpha_gl)
            run_experiment(nodes, "generalization_loss", alpha_gl=alpha_gl, num_repeats=NUM_REPEATS)
        for alpha_pq in PQ_ALPHA_VALUES:
            print("Numero nodi: ", nodes, "Early stopping: progress_quotient Alpha: ", alpha_pq)
            run_experiment(nodes, "progress_quotient", alpha_pq=alpha_pq, num_repeats=NUM_REPEATS)

run()
