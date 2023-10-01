In the pursuit of developing an effective credit card fraud detection model, this project encompassed several crucial steps, from data preparation to model evaluation. Despite not performing hyperparameter tuning, the final model achieved commendable results, reinforcing its potential to safeguard financial transactions and protect consumers. Here are the noteworthy highlights:

1. Data Preparation and Exploration:

    The project commenced with the acquisition of credit card transaction data, which was stored in PostgreSQL.
    The dataset was thoughtfully split into training and testing sets, laying the foundation for subsequent model development.
    Initial data exploration provided valuable insights into the dataset's structure and characteristics.

2. Addressing Class Imbalance:

    Recognizing the inherent class imbalance, characterized by a significantly smaller number of fraudulent transactions, we applied the Synthetic Minority Over-sampling Technique (SMOTE).
    SMOTE facilitated the creation of a balanced training dataset, enabling the model to learn from both fraudulent and non-fraudulent transactions effectively.

3. Model Building:

    A Random Forest classifier, known for its adaptability and versatility, was constructed to tackle the fraud detection challenge.
    The model was trained on the resampled training data, leveraging the benefits of a balanced class distribution.

4. Model Evaluation:

    Model evaluation was centered around the critical metric, Area Under the Precision-Recall Curve (AUPRC), which aptly addresses imbalanced datasets.
    Additional metrics, including precision, recall, and F1-score, were considered, providing a comprehensive assessment of model performance.
    The Precision-Recall curve visually depicted the model's trade-offs between precision and recall.

5. Key Outcomes:

    The final model exhibited notable performance characteristics:
        Precision: Achieving an impressive 0.85 for identifying fraudulent transactions, indicating a high rate of accurate fraud predictions.
        Recall: Successfully identifying 82% of actual fraudulent transactions, considering the inherent challenge of class imbalance.
        F1-Score: Striking a balance between precision and recall, with a score of 0.84 for the positive class.
        AUPRC: Demonstrating a robust performance metric of 0.85, reflecting the model's ability to maintain high precision across various recall levels.
        Accuracy: Although overall accuracy reached 100%, the model's effectiveness was appropriately evaluated using more pertinent metrics.
