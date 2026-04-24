# Machine Learning LabWork-7 Report Part-2: Wildfire
by Sema Nur ALAN \ Group: EDIfu-23 \ ID: 20233635 \ VilniusTech

## Project Overview
This project implements a machine learning solution for classifying wildfire images using computer vision and various classification algorithms. Due to the large dataset size, the implementation includes dataset sampling and evaluation of multiple models.

## Implementation Details

### Data Processing Pipeline
1. **Dataset Sampling**
   - User-defined sampling ratio (Current analysis uses 10% of the dataset)
   - Balanced sampling from both wildfire and non-wildfire classes
   - Original dataset split maintained (train/valid/test)

2. **Image Preprocessing**
   - Image resizing to 64x64 pixels
   - RGB color format
   - Pixel normalization (values scaled to 0-1 range)

3. **Feature Engineering**
   - Standard scaling of flattened image features
   - PCA dimensionality reduction to 100 components
   - Train/test split with stratification

## Models Implemented

Three different classification models were implemented and compared:
1. Support Vector Machine (SVM) with linear kernel
2. Random Forest Classifier (100 estimators)
3. K-Nearest Neighbors (KNN) with k=5

## Model Performance

### Detailed Model Metrics

| Model | Cross-Validation Accuracy | Test Accuracy | Precision | Recall | F1 Score | Specificity |
|-------|-------------------------|---------------|-----------|---------|-----------|-------------|
| SVM | 0.9884 | 0.9931 | 0.9938 | 0.9929 | 0.9933 | 0.9992 |
| Random_Forest | 0.9877 | 0.9940 | 0.9942 | 0.9938 | 0.9939 | 0.9993 |
| KNN | 0.7714 | 0.8691 | 0.8698 | 0.8723 | 0.8694 | 0.9855 |

### Confusion Matrices Analysis
The confusion matrices show the binary classification results (wildfire vs. no-wildfire):

![](plots_wildfire/SVM_confusion_matrix.png)
**SVM Confusion Matrix**: Shows strong performance in both classes with minimal misclassifications.

![](plots_wildfire/Random_Forest_confusion_matrix.png)
**Random Forest Confusion Matrix**: Demonstrates excellent balance between classes and high accuracy.

![](plots_wildfire/KNN_confusion_matrix.png)
**KNN Confusion Matrix**: Shows more misclassifications compared to other models, but still maintains good performance.

### Sample Predictions Analysis
These visualizations show actual test images with their predicted (P) and true (T) labels:

![](plots_wildfire/SVM_sample_predictions.png)
![](plots_wildfire/Random_Forest_sample_predictions.png)
![](plots_wildfire/KNN_sample_predictions.png)

### Performance Analysis

![](plots_wildfire/metrics_comparison.png)
**Metrics Comparison**: This visualization compares key performance metrics across models:
- Both SVM and Random Forest achieve exceptional performance with accuracy >99%
- KNN performs relatively well but lags behind with metrics around 87%
- All models maintain high specificity, crucial for minimizing false wildfire alerts

![](plots_wildfire/training_time_comparison.png)
**Training Time Comparison**: Shows the computational efficiency of each model:
- Training times are measured on 10% of the full dataset
- Relative differences show the scalability characteristics of each algorithm

![](plots_wildfire/roc_curve.png)
**ROC Curves**: Shows the trade-off between true positive rate and false positive rate:
- Both SVM and Random Forest show near-perfect ROC curves
- The high AUC values indicate excellent discrimination ability
- KNN's curve shows good but comparatively lower performance

## Model Deployment
The trained models and necessary preprocessing components have been saved:
- Trained models (SVM, Random Forest, KNN)
- Standard Scaler
- PCA transformer
- Label Encoder

These components can be loaded for making new predictions on unseen wildfire images.

## Conclusions
Based on the performance metrics shown above, we can observe that:
1. Dataset Sampling Impact:
   - Analysis performed on 10% of the full dataset shows robust model performance
   - Results suggest that even with reduced data, models can achieve high accuracy

2. Model Performance:
   - Random Forest achieves the highest accuracy (99.40%) and F1 score (0.9939)
   - SVM performs very similarly with 99.31% accuracy
   - KNN shows lower but still acceptable performance with 86.91% accuracy
   - All models demonstrate high specificity, crucial for reliable wildfire detection

3. Practical Implications:
   - Both Random Forest and SVM are suitable for wildfire detection
   - The choice between them might depend on deployment constraints
   - The high specificity across all models suggests reliable false alarm rates
