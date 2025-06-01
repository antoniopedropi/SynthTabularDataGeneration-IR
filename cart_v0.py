import pandas as pd
import numpy as np
from synthpop import MissingDataHandler, DataProcessor, CARTMethod

from synthpop.metrics import MetricsReport

# # ----- Step 1: Example dataset -----
# np.random.seed(42)

# # Majority data
# majority = pd.DataFrame({
#     'feature1': np.random.normal(5, 1, 950),
#     'feature2': np.random.choice(['A', 'B', 'C'], 950),
#     'target': np.random.normal(30, 5, 950)
# })

# # Minority data
# minority = pd.DataFrame({
#     'feature1': np.random.normal(8, 1, 50),
#     'feature2': np.random.choice(['A', 'B', 'C'], 50),
#     'target': np.random.normal(80, 5, 50)
# })

# df = pd.concat([majority, minority], ignore_index=True)


# df = pd.read_csv('../datasets/abalone.csv', sep=',')
# df.head()

# print("Original target distribution:\n", df['Rings'].describe())

# # ----- Step 2: Define minority threshold -----
# threshold = 15
# minority_df = df[df['Rings'] > threshold].reset_index(drop=True)  # Reset index is IMPORTANT

# print(f"\nNumber of minority samples: {len(minority_df)}")

# # ----- Step 3: Initialize handlers -----
# md_handler = MissingDataHandler()

# # VERY IMPORTANT: Get metadata ONLY from minority data
# metadata = md_handler.get_column_dtypes(minority_df)

# # ----- Step 4: Impute missing values -----
# # (Even though our simulated data has no missing values, we do this to avoid future issues)
# missingness_dict = md_handler.detect_missingness(minority_df)
# real_df = md_handler.apply_imputation(minority_df, missingness_dict)

# # ----- Step 5: Preprocess -----
# processor = DataProcessor(metadata)
# processed_data = processor.preprocess(real_df)

# # ----- Step 6: Check for missing values -----
# print("\nMissing values after preprocessing:")
# print(processed_data.isnull().sum())

# # Sanity check: If any column has NaN, stop
# if processed_data.isnull().any().any():
#     raise ValueError("Missing values found in processed_data — preprocessing failed!")

# # ----- Step 7: Fit CART -----
# cart = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=42)
# cart.fit(processed_data)

# # ----- Step 8: Sample synthetic minority data -----
# n_samples = 500
# synthetic_processed = cart.sample(n_samples)

# # ----- Step 9: Postprocess -----
# synthetic_df = processor.postprocess(synthetic_processed)

# print("\nSynthetic minority data sample:")
# print(synthetic_df.head())

# print("\nSynthetic target distribution:")
# print(synthetic_df['Rings'].describe())

# report = MetricsReport(real_df, synthetic_df, metadata)
# report_df = report.generate_report()
# print(report_df)




def generate_synthetic_minority_data(df, target_column, threshold, n_samples=500, random_state=4040):
    """
    Generate synthetic data for the minority group where target_column > threshold.
    
    Parameters:
    df (pd.DataFrame): Original dataset
    target_column (str): Name of the continuous target variable
    threshold (float): Value to define the minority region
    n_samples (int): Number of synthetic minority samples to generate
    random_state (int): Random state for reproducibility
    
    Returns:
    synthetic_df (pd.DataFrame): Synthetic minority data
    report_df (pd.DataFrame): Utility metrics comparing real vs synthetic minority data
    """

    # ----- Step 1: Select minority data -----
    minority_df = df[df[target_column] > threshold].reset_index(drop=True)
    print(f"Number of minority samples found: {len(minority_df)}")

    if len(minority_df) < 5:
        raise ValueError("Too few minority samples. Consider lowering the threshold.")

    # ----- Step 2: Metadata -----
    md_handler = MissingDataHandler()
    metadata = md_handler.get_column_dtypes(minority_df)

    # ----- Step 3: Impute missing values -----
    missingness_dict = md_handler.detect_missingness(minority_df)
    real_df = md_handler.apply_imputation(minority_df, missingness_dict)

    # ----- Step 4: Preprocess -----
    processor = DataProcessor(metadata)
    processed_data = processor.preprocess(real_df)

    # ----- Step 5: Check for missing values -----
    if processed_data.isnull().any().any():
        raise ValueError("Missing values found in processed_data — preprocessing failed!")

    # ----- Step 6: Fit CART -----
    cart = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=random_state)
    cart.fit(processed_data)

    # ----- Step 7: Sample synthetic minority data -----
    synthetic_processed = cart.sample(n_samples)

    # ----- Step 8: Postprocess -----
    synthetic_df = processor.postprocess(synthetic_processed)

    # ----- Step 9: Metrics -----
    report = MetricsReport(real_df, synthetic_df, metadata)
    report_df = report.generate_report()

    print("\nSynthetic minority target distribution:")
    print(synthetic_df[target_column].describe())
    
    # ----- Step 10: Create augmented dataset -----
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    return augmented_df, synthetic_df, report_df


# df = pd.read_csv('../datasets/abalone.csv')

# augmented_data, synthetic_data, metrics = generate_synthetic_minority_data(
#     df,
#     target_column='Rings',
#     threshold=15,
#     n_samples=500
# )

# print("\nAugmented dataset shape:", augmented_data.shape)
# print("\nSynthetic data sample:")
# print(synthetic_data.head())

# print("\nSimilarity metrics:")
# print(metrics)




