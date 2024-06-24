**Facial Expression Recognition with Feature Extraction**

This repository implements a system for facial expression recognition using machine learning classifiers. It extracts features from facial landmark data (.pts files) and trains various classifiers to predict expressions.

**Features:**

- Loads facial landmark data from a directory structure.
- Extracts features based on landmark point coordinates.
- Trains and evaluates multiple machine learning classifiers.
- Calculates performance metrics (accuracy, confusion matrix, classification report).
- Visualizes confusion matrices for classifier performance analysis.

**Data Format:**

The code expects facial expression data files with the following format:

- Files should be named in the format "m-xxx-yy.pts" or "w-xxx-yy.pts", where:
  - "m" or "w" indicates male or female (optional).
  - "xxx" represents a unique identifier.
  - "yy" represents the expression label (e.g., 1 for neutral, 2 for smile, etc.).
- Each file should contain point coordinates, one per line, in the following format:
- X1 Y1
- X2 Y2
- ...
- Xn Yn
  - X and Y represent the coordinates of a landmark point on the face.

**Getting Started**

1. **Prerequisites:**

- Python 3.x
- NumPy (pip install numpy)
- Scikit-learn (pip install scikit-learn)
- Matplotlib (pip install matplotlib)

2. **Clone the Repository:**

   **git clone https://github.com/csc929/facial-expression-recognition.git**

3. **Data Preparation:**

- Place your facial landmark data (.pts files) in a directory structure with subdirectories for organization.
- Ensure your data files follow the filename format used by the code for label extraction (refer to code comments for details).

4. **Running the Script:**

- Clone the repository (if applicable).
- Replace the placeholder data directory path (data_dir_path) in the \__main__ block with the actual path to your facial landmark data.
- Ensure your data files are organized as expected (subdirectories with .pts files) and follow the filename format for label extraction.
- Run the script (usually python main.py or the script's filename).

**Code Structure:**

- **Imports:** Necessary libraries for data manipulation, machine learning, and visualization.
- **ExpressionIdentifier Class:**
  - \__init_\_: Initializes the object with the data directory path and pre-loads features and labels during initialization.
  - load_data: Loads facial landmark data from all .pts files within subdirectories, extracts features, and creates labels based on filenames.
  - evaluate_classifier: Trains a specified classifier, evaluates performance on test data, and returns metrics.
  - visualize_confusion_matrix: Visualizes a confusion matrix using Matplotlib.
- **Main Script:**
  - Defines the data directory path.
  - Creates an ExpressionIdentifier object.
  - Loops through a list of classifiers (KNN, Naive Bayes, Decision Tree, Neural Network, SVM).
    - Trains and evaluates each classifier.
    - Prints accuracy, confusion matrix, and classification report.
    - Visualizes the confusion matrix for each classifier.

**Explanation of the Code:**

- ExpressionIdentifier **Class:** This class handles data loading, feature extraction, classifier training, evaluation, and confusion matrix visualization.
- **Data Loading and Feature Extraction:** The code loads facial landmark data from .pts files and extracts seven features based on landmark point coordinates.
- **Classifier Training and Evaluation:** Different machine learning classifiers are trained on the extracted features and evaluated using metrics like accuracy, confusion matrix, and classification report.
- **Confusion Matrix Visualization:** The code utilizes Matplotlib to visualize the confusion matrix for each classifier, providing insights into model performance.

**Expected Outputs:**

The script will print the evaluation results (accuracy, confusion matrix, classification report) for each classifier. Additionally, it will visualize the confusion matrix for each classifier.

**Further Development:**

- Implement unit tests for code robustness.
- Explore additional feature extraction techniques and machine learning models.
- Integrate with a real-time facial landmark detection system (if applicable).

**Note:**

- This code focuses on core functionalities and might not include extensive error handling or data cleaning routines.
- The chosen facial expression features and hyperparameter configurations for classifiers can be customized based on your specific dataset and needs.

This README provides a basic overview of the code. Refer to the code comments for further details and consider adding unit tests for improved robustness.
