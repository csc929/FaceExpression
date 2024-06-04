import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ExpressionIdentifier:
    """
    Class for loading facial expression data, training classifiers, and evaluating performance.
    """

    def __init__(self, data_dir):
        """
        Initializes the ExpressionIdentifier object with data directory path.
        """
        self.data_dir = data_dir
        self.label_mapping = {1: 'neutral', 2: 'smile', 3: 'anger', 5: 'left light on'}
        self.features, self.labels = self.load_data()  # Load data during initialization

    def load_data(self):
        """
        Loads data from all .pts files within subdirectories in the given parent directory.

        Returns:
            tuple: A tuple containing:
                - features (list): List of extracted features for each data point.
                - labels (list): List of corresponding expression labels.
        """

        features = []
        labels = []

        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.pts'):
                        # Extract features and label from data file
                        filepath = os.path.join(subdir_path, filename)
                        with open(filepath, 'r') as f:
                            lines = f.readlines()

                            points = np.array(
                                [[float(x) for x in line.split()] for line in lines[3:-1]])  # Skip header and last line

                            # Calculate features based on point coordinates
                            eye_length_ratio = max(np.linalg.norm(points[1] - points[0]),
                                                   np.linalg.norm(points[4] - points[5])) / np.linalg.norm(
                                points[7] - points[12])
                            eye_distance_ratio = np.linalg.norm(points[0] - points[3]) / np.linalg.norm(
                                points[7] - points[12])
                            nose_ratio = np.linalg.norm(points[14] - points[15]) / np.linalg.norm(
                                points[19] - points[20])
                            lip_size_ratio = np.linalg.norm(points[1] - points[2]) / np.linalg.norm(
                                points[16] - points[17])
                            lip_length_ratio = np.linalg.norm(points[1] - points[2]) / np.linalg.norm(
                                points[19] - points[20])
                            eyebrow_length_ratio = max(np.linalg.norm(points[3] - points[4]),
                                                       np.linalg.norm(points[5] - points[6])) / np.linalg.norm(
                                points[7] - points[12])
                            aggressive_ratio = np.linalg.norm(points[9] - points[18]) / np.linalg.norm(
                                points[19] - points[20])

                            features.append([eye_length_ratio, eye_distance_ratio, nose_ratio,
                                             lip_size_ratio, lip_length_ratio, eyebrow_length_ratio, aggressive_ratio])

                            # Extract label from filename (format "m-xxx-yy.pts" or "w-xxx-yy.pts")
                            label_parts = filename.split('-')
                            label = int(label_parts[-1].split('.')[0])
                            labels.append(self.label_mapping[label])

        return features, labels

    def evaluate_classifier(self, classifier_name, **kwargs):
        """
        Trains a classifier, predicts on test data, and evaluates performance.

        Args:
        classifier_name (str): Name of the classifier being evaluated.
        **kwargs (dict): Optional keyword arguments for specific classifier hyperparameters.

        Returns:
            tuple: A tuple containing:
                - classifier: The trained classifier object.
                - accuracy (float): Overall accuracy of the classifier.
                - confusion_mtrx (np.ndarray): Confusion matrix for the classifier.
                - classification_rpt (str): Classification report for the classifier.
        """
        # Split data into training and testing sets (80/20 split)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

        classifier = None

        # Define default hyperparameter configurations for each classifier
        classifier_params = {
            'KNN': {'n_neighbors': 5},
            'Naive Bayes': {},
            'Decision Tree': {},
            'ANN': {'solver': 'lbfgs', 'alpha': 1e-5, 'hidden_layer_sizes': (10, 2), 'random_state': 1},
            'SVM': {},
        }

        # Update hyperparameters with kwargs (if provided)
        if classifier_name in classifier_params:
            classifier_params[classifier_name].update(kwargs)

            # Initialize classifier with hyperparameters from kwargs
            if classifier_name == 'KNN':
                classifier = KNeighborsClassifier(**classifier_params[classifier_name])
            elif classifier_name == 'Naive Bayes':
                classifier = GaussianNB()
            elif classifier_name == 'Decision Tree':
                classifier = DecisionTreeClassifier(**classifier_params[classifier_name])
            elif classifier_name == 'ANN':
                classifier = MLPClassifier(**classifier_params[classifier_name])
            elif classifier_name == 'SVM':
                classifier = SVC(**classifier_params[classifier_name])

        # Train the classifier
        classifier.fit(x_train, y_train)

        # Predict on test data
        y_pred = classifier.predict(x_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        confusion_mtrx = confusion_matrix(y_test, y_pred)
        classification_rpt = classification_report(y_test, y_pred)

        return classifier, accuracy, confusion_mtrx, classification_rpt

    def visualize_confusion_matrix(self, confusion_matrix_, classifier_name):
        """
        Visualizes the confusion matrix for a given classifier.

        Args:
            confusion_matrix_ (np.ndarray): Confusion matrix to visualize.
            classifier_name (str): Name of the classifier for the confusion matrix.
        """

        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix_, cmap='Blues')
        plt.colorbar()
        classes = list(self.label_mapping.values())
        plt.xticks(np.arange(len(classes)), classes, rotation=45)
        plt.yticks(np.arange(len(classes)), classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for {classifier_name}')
        plt.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(confusion_matrix_[i, j]), ha='center', va='center', fontsize=8)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    data_dir_path = 'c:/Projects/faceExpression/data/points_22'

    # Create ExpressionIdentifier object
    identifier = ExpressionIdentifier(data_dir_path)

    # Evaluate classifiers
    classifiers = ['KNN', 'Naive Bayes', 'Decision Tree', 'ANN', 'SVM']
    for classifier_type_name in classifiers:
        print(f"\nEvaluating {classifier_type_name} classifier:")
        _classifier, _accuracy, _confusion_matrix, _classification_report = identifier.evaluate_classifier(
            classifier_type_name)
        print(f"\nAccuracy: {_accuracy:.4f}")
        print(f"\nConfusion Matrix:\n{_confusion_matrix}")
        print(f"\nClassification Report:\n{_classification_report}")

        # Visualize confusion matrix for each classifier
        identifier.visualize_confusion_matrix(_confusion_matrix, classifier_type_name)
