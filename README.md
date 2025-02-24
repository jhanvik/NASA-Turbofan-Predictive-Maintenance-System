# **NASA Turbofan Predictive Maintenance System**
**Predictive Maintenance Model for Turbofan Engines using LSTM**

## **Overview**
This project focuses on developing a predictive maintenance model for NASA's turbofan engines. The primary objective is to accurately predict the Remaining Useful Life (RUL) of these engines, optimizing maintenance schedules and reducing the likelihood of unexpected failures. The project leverages advanced machine learning techniques, particularly Long Short-Term Memory (LSTM) networks, to analyze time-series data.

## **Dataset**
We utilized the NASA C-MAPSS dataset, which provides detailed sensor measurements from turbofan engines under various operating conditions. Specifically, we focused on the FD002 subset of the dataset. The dataset includes:
- Training and testing data files with multiple sensors and operational settings.
- Ground truth RUL values for evaluation.
- **Dataset Link:** https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## **Data Loading and Preprocessing**
The data loading phase involved reading the training, testing, and RUL data files. We then performed extensive preprocessing to ensure the data's quality and relevance:
- **Normalization:** Sensor data were normalized to eliminate discrepancies in measurement units and scales.
- **Sensor Selection:** We selected significant sensors based on their variance to focus on those contributing most to RUL predictions.
- **RUL Calculation:** We calculated the RUL for each engine cycle in the training data.

## **Sequence Generation**
To prepare the data for LSTM training, we generated sequences of fixed length, ensuring the model could learn temporal dependencies in the data. Sequences were created from the normalized sensor data, with corresponding RUL values as targets.

## **Model Architecture**
The predictive maintenance model was built using an LSTM-based neural network. The architecture included:
- Multiple LSTM layers to capture long-term dependencies.
- Batch normalization and dropout layers to improve convergence and prevent overfitting.
- Fully connected layers and a regression layer to produce the final RUL prediction.

## **Training Configuration**
The model was trained using the Adam optimizer with a dynamic learning rate schedule and early stopping criteria. Training involved multiple epochs to allow the model to learn complex patterns in the data while preventing overfitting through validation.

## **Training Progress**
The following image illustrates the training progress of the model, focusing on the Root Mean Squared Error (RMSE) and Loss over iterations and epochs:

![Training Progress](https://github.com/user-attachments/assets/5d8988d4-399a-44c7-97a4-5269cf80f6f3)


The image shows the training progress of the machine learning model. The top graph displays the RMSE for both training and validation data, while the bottom graph shows the Loss for the training data. The graphs indicate how the RMSE and Loss decrease over time, suggesting the model is learning and improving its performance.

## **Model Evaluation**
The following image illustrates the evaluation of the model's performance on the training and validation datasets:

![Evaluation Results](https://github.com/user-attachments/assets/cfe82620-9271-4120-acb7-4146b67c3498)



- **Left Plot (Training Predictions):** Shows a comparison between Actual RUL (blue line) and Predicted RUL (orange area) over a range of samples. The x-axis represents the sample number, and the y-axis represents the RUL.
- **Right Plot (Validation Predictions):** A scatter plot comparing Predicted RUL to Actual RUL. The x-axis represents Actual RUL, and the y-axis represents Predicted RUL. The red dashed line represents the ideal prediction line where Predicted RUL equals Actual RUL. The correlation coefficient (R = 0.86) indicates a strong positive correlation between the actual and predicted values.

## **Actual vs. Predicted RUL on Test Data**
The following image illustrates the model's performance on the test dataset:

![Actual vs. Predicted RUL on Test Data](https://github.com/user-attachments/assets/69f75805-9af5-4ab7-b7dd-66503a71515e)


 The image is a scatter plot titled "Actual vs. Predicted RUL on Test Data." The x-axis is labeled "Actual RUL" and ranges from 0 to 200, while the y-axis is labeled "Predicted RUL" and ranges from 0 to 160. The plot contains numerous blue dots representing data points, with a red dashed line indicating the ideal scenario where the predicted RUL equals the actual RUL. This plot visually compares the predicted Remaining Useful Life (RUL) of a system or component against the actual RUL, highlighting the accuracy and performance of the predictive model.

## **Model Evaluation Metrics**
We evaluated the model's performance on both the validation and test datasets using several metrics:
- **Root Mean Square Error (RMSE):** Measures the average magnitude of prediction errors.
- **Mean Absolute Error (MAE):** Provides the average absolute error between predicted and actual RUL values.
- **R-Squared (R²):** Indicates the proportion of variance in the RUL that is predictable from the sensor data.

## **Results**
- **Training Performance:** The model showed significant improvement in RMSE during training, demonstrating its learning capability.
- **Validation Performance:** The model achieved a strong correlation coefficient (R = 0.86), indicating good predictive performance on validation data.
- **Test Performance:** The model's test RMSE and MAE were relatively high, and the negative R-Squared value indicated potential overfitting and issues with generalization.

## **Conclusion**
While the model demonstrated strong performance on training and validation datasets, the test results indicated a need for further improvements. Future work could involve additional regularization techniques, more robust preprocessing, or incorporating more training data to enhance generalization.

## **Future Work**
Potential future enhancements could include:
- Incorporating additional data subsets for broader generalization.
- Exploring alternative deep learning architectures and techniques.
- Implementing real-time prediction capabilities for practical applications.
