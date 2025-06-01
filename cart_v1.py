import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
from synthpop import MissingDataHandler, DataProcessor, CARTMethod
from synthpop.metrics import MetricsReport

class RarityWeightedCARTSynthesizer:
    def __init__(self, df, target_column, random_state=42):
        self.df = df.reset_index(drop=True)
        self.target_column = target_column
        self.random_state = random_state

    def _compute_global_rarity(self):
        target_values = self.df[self.target_column].values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
        kde.fit(target_values)
        density = np.exp(kde.score_samples(target_values))
        rarity = 1 / (density + 1e-5)
        rarity = rarity / rarity.sum()
        self.df = self.df.copy()
        self.df['global_rarity'] = rarity

    def generate_synthetic_data(self, n_samples_total=500, resample_size=None):
        self._compute_global_rarity()

        # ----- Step 1: Rarity-weighted resampling -----
        if resample_size is None:
            resample_size = len(self.df)

        # Resample WITHOUT resetting index — keep track of original rows
        resampled_df = self.df.sample(
            n=resample_size,
            replace=True,
            weights=self.df['global_rarity'],
            random_state=self.random_state
        )
        
        # Count how many times each original row was sampled
        self.df['resample_count'] = 0
        resample_indices = resampled_df.index.value_counts()
        self.df.loc[resample_indices.index, 'resample_count'] = resample_indices.values
        
        # NOW reset index for downstream processing
        resampled_df = resampled_df.reset_index(drop=True)
        
        self.resampled_df = resampled_df  # Save for later plotting

        
        print(f"Resampled {len(resampled_df)} rows based on rarity weighting.")


        # ----- Step 2: Fit CART on the resampled data -----
        
        # Drop metadata columns from resampled data before fitting the model
        resampled_df_for_model = resampled_df.drop(columns=['global_rarity', 'resample_count'], errors='ignore')
        
        md_handler = MissingDataHandler()
        metadata = md_handler.get_column_dtypes(resampled_df_for_model)
        missingness_dict = md_handler.detect_missingness(resampled_df_for_model)
        real_df = md_handler.apply_imputation(resampled_df_for_model, missingness_dict)
        
        processor = DataProcessor(metadata)
        processed_data = processor.preprocess(real_df)
        
        if processed_data.isnull().any().any():
            raise ValueError("Missing values found in processed_data — preprocessing failed!")
        
        cart = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=self.random_state)
        cart.fit(processed_data)

        # md_handler = MissingDataHandler()
        # metadata = md_handler.get_column_dtypes(resampled_df)
        # missingness_dict = md_handler.detect_missingness(resampled_df)
        # real_df = md_handler.apply_imputation(resampled_df, missingness_dict)

        # processor = DataProcessor(metadata)
        # processed_data = processor.preprocess(real_df)

        # if processed_data.isnull().any().any():
        #     raise ValueError("Missing values found in processed_data — preprocessing failed!")

        # cart = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=self.random_state)
        # cart.fit(processed_data)

        # ----- Step 3: Sample synthetic data -----
        synthetic_processed = cart.sample(n_samples_total)
        synthetic_df = processor.postprocess(synthetic_processed)

        self.synthetic_df = synthetic_df
        #self.augmented_df = pd.concat([self.df.drop(columns='global_rarity'), synthetic_df], ignore_index=True)
        
        # Create a copy of the original data
        original_df = self.df.copy()
        original_df['origin'] = 'real'
        
        # Create synthetic data copy with matching metadata columns set to NaN
        synthetic_df_with_meta = synthetic_df.copy()
        synthetic_df_with_meta['resample_count'] = np.nan
        synthetic_df_with_meta['global_rarity'] = np.nan
        synthetic_df_with_meta['origin'] = 'synthetic'
        
        # Concatenate
        self.augmented_df = pd.concat([original_df, synthetic_df_with_meta], ignore_index=True)

        return self.augmented_df, synthetic_df

    def plot_distributions(self, show_rarity=False, compare_resampled=True):
        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to plot.")
            return

        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.df[self.target_column], label='Real Data', fill=True, alpha=0.5)
        sns.kdeplot(self.synthetic_df[self.target_column], label='Synthetic Data', fill=True, alpha=0.5)
        plt.title('Target Variable Distribution: Real vs Synthetic Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        if compare_resampled and hasattr(self, 'synthetic_df') and hasattr(self, 'df') and 'resample_count' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.df[self.target_column], label='Original Data', fill=True, alpha=0.5)
            sns.kdeplot(self.resampled_df[self.target_column], label='Resampled Data', fill=True, alpha=0.5)
            plt.title('Original vs Resampled Target Distribution')
            plt.xlabel(self.target_column)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()

        # New plot: resample count vs target
        if 'resample_count' in self.df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[self.target_column], self.df['resample_count'], alpha=0.5)
            plt.title('Resample Count vs Target Value')
            plt.xlabel(self.target_column)
            plt.ylabel('Resample Count')
            plt.grid(True)
            plt.show()

        if show_rarity:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df[self.target_column], self.df['global_rarity'], alpha=0.5, label='Global Rarity')
            plt.title('Global Rarity Scores by Target Value')
            plt.xlabel(self.target_column)
            plt.ylabel('Global Rarity')
            plt.legend()
            plt.grid(True)
            plt.show()
            
    def plot_augmented_vs_original(self):
        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to plot.")
            return
    
        if self.augmented_df is None or self.augmented_df.empty:
            print("No augmented data to plot.")
            return
    
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.df[self.target_column], label='Original Data', fill=True, alpha=0.5)
        sns.kdeplot(self.augmented_df[self.target_column], label='Augmented Data (Original + Synthetic)', fill=True, alpha=0.5)
        plt.title('Target Variable Distribution: Original vs Augmented Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
    
        # Optional: overlay histogram too
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[self.target_column], label='Original Data', color='blue', alpha=0.5, kde=False, stat='density')
        sns.histplot(self.augmented_df[self.target_column], label='Augmented Data', color='orange', alpha=0.5, kde=False, stat='density')
        plt.title('Histogram: Original vs Augmented Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_metrics(self):
        if self.synthetic_df is None or self.synthetic_df.empty:
            print("No synthetic data to evaluate.")
            return None

        md_handler = MissingDataHandler()
        common_cols = list(set(self.df.columns) & set(self.synthetic_df.columns))
        real_df = self.df[common_cols]
        synthetic_df = self.synthetic_df[common_cols]

        metadata = md_handler.get_column_dtypes(real_df)
        missingness_dict = md_handler.detect_missingness(real_df)
        real_df_clean = md_handler.apply_imputation(real_df, missingness_dict)

        missingness_dict_syn = md_handler.detect_missingness(synthetic_df)
        synthetic_df_clean = md_handler.apply_imputation(synthetic_df, missingness_dict_syn)

        report = MetricsReport(real_df_clean, synthetic_df_clean, metadata)
        return report.generate_report()



# Load data
df = pd.read_csv("../datasets/abalone.csv")

# Instantiate synthesizer
synth = RarityWeightedCARTSynthesizer(df, target_column='Rings')

# Generate synthetic data (can adjust resample_size if desired)
augmented_df, synthetic_df = synth.generate_synthetic_data(
    n_samples_total=2000,
    resample_size=200  # Optional
)

# Plot distributions and rarity
synth.plot_distributions(show_rarity=True, compare_resampled=True)

synth.plot_augmented_vs_original()

# Check similarity metrics
metrics = synth.generate_metrics()
print(metrics)


