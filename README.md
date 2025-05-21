# This repo is intended for applying EEGNET to EEG Data

One source for this EEG data is https://openneuro.org/search/modality/eeg?query=%7B%22modality_selected%22%3A%22eeg%22%7D.

# EEG Classification with EEGNet

This project implements EEG signal classification using different configurations of the EEGNet neural network model. It includes scripts for defining the models, notebooks for running inter-subject and intra-subject classification experiments, and a notebook for plotting training and validation metrics from TensorBoard logs.

## Project Structure

- `eegnet.py`: Contains the Python code defining various EEGNet model architectures (e.g., EEGNet_4_2, EEGNet_8_2).
- `inter_subject.ipynb`: Jupyter Notebook for inter-subject EEG classification experiments.
- `intra_subject.ipynb`: Jupyter Notebook for intra-subject EEG classification experiments.
- `plotting.ipynb`: Jupyter Notebook for visualizing training and validation metrics from TensorBoard logs.
- `training_results/`: Directory likely containing TensorBoard log files from experiments (as referenced in `plotting.ipynb`).
- `saved_plots/`: Directory where generated plots from `plotting.ipynb` are saved.

## Models

The `eegnet.py` script defines the following EEGNet configurations:

- EEGNet with 4 temporal filters and 2 spatial filters (`EEGNet_4_2`)
- EEGNet with 8 temporal filters and 2 spatial filters (`EEGNet_8_2`)
- EEGNet with 16 temporal filters and 2 spatial filters (`EEGNet_16_2`)
- EEGNet with 16 temporal filters and 4 spatial filters (`EEGNet_16_4`)

These models utilize standard, depthwise, and separable convolutions, along with batch normalization, ELU activation, average pooling, and dropout for classification tasks.

## Experiments

The project supports two main types of EEG classification experiments:

### Inter-Subject Classification (`inter_subject.ipynb`)

This notebook is designed to train and evaluate models on data where the training and testing sets come from different subjects. This assesses the model's ability to generalize across individuals.

### Intra-Subject Classification (`intra_subject.ipynb`)

This notebook focuses on training and evaluating models for individual subjects, where the training and testing data come from the same subject. This is often used for personalized BCI applications.

## Plotting and Results

The `plotting.ipynb` notebook provides tools to:

- Load training and validation metrics (Loss, Accuracy, Precision, Recall, F1-Score, ROC_AUC) from TensorBoard event files.
- Generate and save plots comparing training and validation performance over epochs.

## Dependencies

The project relies on the following Python libraries:

- numpy
- torch (PyTorch)
- pandas
- tensorboard
- matplotlib
- os

## Setup and Usage

1.  **Prerequisites**: Ensure you have Python and the listed dependencies installed.
2.  **Data**: Prepare your EEG data. The notebooks (`inter_subject.ipynb` and `intra_subject.ipynb`) will need to be adapted to load your specific data format.
3.  **Training**:
    - Modify the data loading and preprocessing sections in the relevant Jupyter Notebook (`inter_subject.ipynb` or `intra_subject.ipynb`) to suit your dataset.
    - Choose an EEGNet configuration from `eegnet.py` or define your own within the notebook.
    - Run the notebook cells to train the model. Ensure TensorBoard logging is active during training to generate logs for plotting.
4.  **Plotting**:
    - Update the `log_dir` variable in `plotting.ipynb` to point to your TensorBoard log directory.
    - Run the notebook to generate and save performance plots.
