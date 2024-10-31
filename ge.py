import numpy as np
import random
import matplotlib.pyplot as plt

# Load and preprocess the WDBC dataset
def load_wdbc_data(file_path):
    dataset = []
    with open(file_path, 'r') as data_file:
        for line in data_file:
            elements = line.strip().split(',')
            features = list(map(float, elements[2:]))  # Extract features
            label = 1 if elements[1] == 'M' else 0  # Label encoding
            dataset.append((features, label))
    return dataset

# Feature normalization function
def normalize_features(data):
    feature_matrix = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    normalized_features = (feature_matrix - feature_matrix.mean(axis=0)) / feature_matrix.std(axis=0)
    return normalized_features, labels

# Create k-fold splits
def create_k_folds(data, num_folds=10):
    random.shuffle(data)
    fold_size = len(data) // num_folds
    return [data[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

# ReLU activation function
def relu_activation(z):
    return np.maximum(0, z)

# Softmax activation function
def softmax_activation(z):
    exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Neural Network Class Definition
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_dim = input_size
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_size
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1 for i in range(len(self.layers) - 1)]
        self.biases = [np.random.randn(1, self.layers[i + 1]) * 0.1 for i in range(len(self.layers) - 1)]

    def forward_pass(self, inputs):
        self.activations = [inputs]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(relu_activation(z))  # Use ReLU for hidden layers
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.output = softmax_activation(z)  # Softmax for output layer
        return self.output

    def predict(self, inputs):
        probabilities = self.forward_pass(inputs)
        return np.argmax(probabilities, axis=1)

# Genetic Algorithm for Neural Network Optimization
class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, generations):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def initialize_population(self, input_dim, hidden_layers, output_dim):
        return [MultiLayerPerceptron(input_dim, hidden_layers, output_dim) for _ in range(self.pop_size)]

    def crossover(self, parent_a, parent_b):
        offspring = MultiLayerPerceptron(parent_a.input_dim, parent_a.hidden_sizes, parent_a.output_dim)
        for i in range(len(parent_a.weights)):
            mask = np.random.rand(*parent_a.weights[i].shape) > 0.5
            offspring.weights[i] = np.where(mask, parent_a.weights[i], parent_b.weights[i])
        return offspring

    def apply_mutation(self, nn):
        for i in range(len(nn.weights)):
            if np.random.rand() < self.mutation_rate:
                nn.weights[i] += np.random.randn(*nn.weights[i].shape) * 0.1

    def calculate_fitness(self, nn, features, targets):
        predictions = nn.predict(features)
        return np.mean(predictions == targets)

    def optimize(self, train_features, train_labels, val_features, val_labels, hidden_layers):
        population = self.initialize_population(train_features.shape[1], hidden_layers, 2)
        for generation in range(self.generations):
            population = sorted(population, key=lambda nn: self.calculate_fitness(nn, val_features, val_labels), reverse=True)
            best_fitness = self.calculate_fitness(population[0], val_features, val_labels)
            print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {best_fitness:.4f}")
            selected_parents = population[:self.pop_size // 2]
            for _ in range(self.pop_size // 2):
                parent_a, parent_b = random.sample(selected_parents, 2)
                child = self.crossover(parent_a, parent_b)
                self.apply_mutation(child)
                population.append(child)
        return population[0]

# K-Fold Cross-Validation Function with Fitness Tracking
def k_fold_cross_validation(dataset, num_folds=10, hidden_layers=[45, 30]):
    folds = create_k_folds(dataset, num_folds)
    accuracy_scores = []
    best_fitness_per_fold = []  # To track the best fitness from each fold

    for i in range(num_folds):
        validation_fold = folds[i]
        training_folds = [data for j in range(num_folds) if j != i for data in folds[j]]

        train_features, train_labels = normalize_features(training_folds)
        val_features, val_labels = normalize_features(validation_fold)

        genetic_optimizer = GeneticOptimizer(population_size=30, mutation_rate=0.01, generations=100)
        best_model = genetic_optimizer.optimize(train_features, train_labels, val_features, val_labels, hidden_layers=hidden_layers)

        # Track the best fitness for this fold
        best_fitness = genetic_optimizer.calculate_fitness(best_model, val_features, val_labels)
        best_fitness_per_fold.append(best_fitness)  # Append to list
        accuracy_scores.append(best_fitness)  # Also track as accuracy

    # Print the best fitness for each fold
    for fold_number, fitness in enumerate(best_fitness_per_fold, start=1):
        print(f"Fold {fold_number}/{num_folds}, Best Fitness: {fitness:.4f}")

    avg_accuracy = np.mean(accuracy_scores)
    print("Average Accuracy:", avg_accuracy)

    # Plot best fitness for each fold
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_folds + 1), best_fitness_per_fold, marker='o', linestyle='-')
    plt.xlabel("Fold")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness per Fold in K-Fold Cross Validation")
    plt.grid()
    plt.show()

    return accuracy_scores, avg_accuracy, genetic_optimizer, best_model

# Confusion Matrix Visualization without sklearn
def visualize_confusion_matrix_manual(best_model, features, labels):
    predictions = best_model.predict(features)
    
    tp = fp = tn = fn = 0
    
    for true, pred in zip(labels, predictions):
        if true == 1 and pred == 1:
            tp += 1  # True Positive
        elif true == 1 and pred == 0:
            fn += 1  # False Negative
        elif true == 0 and pred == 0:
            tn += 1  # True Negative
        elif true == 0 and pred == 1:
            fp += 1  # False Positive

    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confusion_matrix, cmap=plt.cm.Oranges, alpha=0.6)
    
    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', fontsize=14)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()

# Main execution block
if __name__ == '__main__':
    dataset_path = 'wdbc.data.txt'
    data = load_wdbc_data(dataset_path)
    accuracy_scores, mean_accuracy, genetic_optimizer, optimal_model = k_fold_cross_validation(data, hidden_layers=[45,30,15])

    # Re-normalize and visualize the confusion matrix manually
    X, Y = normalize_features(data)
    visualize_confusion_matrix_manual(optimal_model, X, Y)
