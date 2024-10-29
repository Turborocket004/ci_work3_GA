import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Sigmoid activation function
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

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
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.1 for i in range(len(self.layers) - 1)]
        self.biases = [np.random.randn(1, self.layers[i + 1]) * 0.1 for i in range(len(self.layers) - 1)]
    
    def forward_pass(self, inputs):
        self.activations = [inputs]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(sigmoid_activation(z))
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.output = softmax_activation(z)
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
        self.performance_history = []

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
    
    def optimize(self, train_features, train_labels, val_features, val_labels):
        population = self.initialize_population(train_features.shape[1], [40, 25], 2)
        for generation in range(self.generations):
            population = sorted(population, key=lambda nn: self.calculate_fitness(nn, val_features, val_labels), reverse=True)
            b_fitness = self.calculate_fitness(population[0], val_features, val_labels)
            self.performance_history.append(b_fitness)
            print(f"Generation {generation + 1}/{self.generations}, Best Fitness: {b_fitness:.4f}")
            selected_parents = population[:self.pop_size // 2]
            for _ in range(self.pop_size // 2):
                parent_a, parent_b = random.sample(selected_parents, 2)
                child = self.crossover(parent_a, parent_b)
                self.apply_mutation(child)
                population.append(child)
        return population[0]
    
    def plot_progress(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.performance_history, marker='o', linestyle='-', color='blue')
        plt.fill_between(range(len(self.performance_history)), self.performance_history, color='lightblue', alpha=0.5)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness', fontsize=12)
        plt.title('Genetic Algorithm Optimization Progress', fontsize=14)
        plt.grid(True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()

# K-Fold Cross-Validation Function
def k_fold_cross_validation(dataset, num_folds=10):
    folds = create_k_folds(dataset, num_folds)
    accuracy_scores = []
    
    # Loop through each fold
    for i in range(num_folds):
        validation_fold = folds[i]
        training_folds = [data for j in range(num_folds) if j != i for data in folds[j]]
        
        train_features, train_labels = normalize_features(training_folds)
        val_features, val_labels = normalize_features(validation_fold)
        
        # Initialize and optimize with Genetic Algorithm
        genetic_optimizer = GeneticOptimizer(population_size=20, mutation_rate=0.01, generations=100)
        best_model = genetic_optimizer.optimize(train_features, train_labels, val_features, val_labels)
        
        # Calculate and store accuracy for the fold
        accuracy = genetic_optimizer.calculate_fitness(best_model, val_features, val_labels)
        accuracy_scores.append(accuracy)
    
    # Print all fold accuracies at the end
    for i, accuracy in enumerate(accuracy_scores, start=1):
        print(f"{i}. Fold Accuracy: {accuracy:.4f}")
    
    # Calculate and display the overall average accuracy
    avg_accuracy = np.mean(accuracy_scores)
    print("Average Accuracy:", avg_accuracy)
    
    return accuracy_scores, avg_accuracy, genetic_optimizer, best_model


# Plotting function for accuracies
def plot_accuracy_with_mean(accuracies, mean_accuracy):
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(accuracies) + 1), accuracies, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axhline(mean_accuracy, color='red', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.4f}')
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('K-Fold Validation Accuracies with Mean', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Confusion Matrix Visualization
def visualize_confusion_matrix(best_model, features, labels):
    predictions = best_model.predict(features)
    confusion_mat = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Oranges)
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()

# Main execution block
if __name__ == '__main__':
    dataset_path = 'wdbc.data.txt'
    data = load_wdbc_data(dataset_path)
    accuracy_scores, mean_accuracy, genetic_optimizer, optimal_model = k_fold_cross_validation(data)
    
    # Plot accuracy scores and mean accuracy
    plot_accuracy_with_mean(accuracy_scores, mean_accuracy)

    # Show GA optimization progress
    genetic_optimizer.plot_progress()

    # Re-normalize and visualize the confusion matrix
    X, Y = normalize_features(data)
    visualize_confusion_matrix(optimal_model, X, Y)
