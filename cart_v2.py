import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns
from synthpop import MissingDataHandler, DataProcessor, CARTMethod
from synthpop.metrics import MetricsReport

class RarityWeightedCARTSynthesizer:
    def __init__(self, df, target_column, random_state=4040):
        self.df = df.reset_index(drop=True)
        self.target_column = target_column
        self.random_state = random_state

    def _compute_global_rarity(self, bandwidth=0.5, alpha=1.5):
        target_values = self.df[self.target_column].values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(target_values)
        density = np.exp(kde.score_samples(target_values))
        rarity = 1 / (density + 1e-5) ** alpha
        rarity = rarity / rarity.sum()
        self.df = self.df.copy()
        self.df['global_rarity'] = rarity

    def generate_synthetic_data(self, n_samples_total=500, resample_size=None):
        self._compute_global_rarity()

        # ----- Step 1: Rarity-weighted resampling -----
        if resample_size is None:
            resample_size = len(self.df)

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

        # Keep a copy for plotting
        self.resampled_df = resampled_df.reset_index(drop=True)

        print(f"Resampled {len(resampled_df)} rows based on rarity weighting.")

        # ----- Step 2: Prepare data for CART -----
        # Drop metadata columns BEFORE fitting the model
        resampled_df_for_model = resampled_df.drop(columns=['global_rarity', 'resample_count'], errors='ignore')
        
        # CRITICAL: reset index to ensure unique indices
        resampled_df_for_model = resampled_df_for_model.reset_index(drop=True)

        md_handler = MissingDataHandler()
        metadata = md_handler.get_column_dtypes(resampled_df_for_model)
        missingness_dict = md_handler.detect_missingness(resampled_df_for_model)
        real_df = md_handler.apply_imputation(resampled_df_for_model, missingness_dict)

        processor = DataProcessor(metadata)
        processed_data = processor.preprocess(real_df)

        if processed_data.isnull().any().any():
            raise ValueError("Missing values found in processed_data — preprocessing failed!")

        # ----- Step 3: Fit CART -----
        cart = CARTMethod(metadata, smoothing=True, proper=True, minibucket=5, random_state=self.random_state)
        cart.fit(processed_data)

        # ----- Step 4: Generate synthetic data -----
        synthetic_processed = cart.sample(n_samples_total)
        synthetic_df = processor.postprocess(synthetic_processed)

        # ----- Step 5: Clean synthetic data -----
        # Ensure metadata columns are removed or set properly
        for col in ['global_rarity', 'resample_count']:
            if col in synthetic_df.columns:
                synthetic_df = synthetic_df.drop(columns=col)

        synthetic_df['resample_count'] = np.nan
        synthetic_df['global_rarity'] = np.nan
        synthetic_df['origin'] = 'synthetic'

        # ----- Step 6: Prepare augmented data -----
        original_df = self.df.copy()
        original_df['origin'] = 'real'

        self.synthetic_df = synthetic_df
        self.augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)

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

        if compare_resampled and hasattr(self, 'resampled_df'):
            plt.figure(figsize=(10, 6))
            sns.kdeplot(self.df[self.target_column], label='Original Data', fill=True, alpha=0.5)
            sns.kdeplot(self.resampled_df[self.target_column], label='Resampled Data', fill=True, alpha=0.5)
            plt.title('Original vs Resampled Target Distribution')
            plt.xlabel(self.target_column)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()

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
        sns.kdeplot(
            data=self.augmented_df, 
            x=self.target_column, 
            hue="origin", 
            fill=True, 
            common_norm=False,
            alpha=0.5
        )
        plt.title('Target Variable Distribution: Original vs Augmented Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.augmented_df, 
            x=self.target_column, 
            hue="origin", 
            stat="density", 
            common_norm=False, 
            multiple="layer", 
            alpha=0.5
        )
        plt.title('Histogram: Original vs Augmented Data')
        plt.xlabel(self.target_column)
        plt.ylabel('Density')
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

# Generate synthetic data
augmented_df, synthetic_df = synth.generate_synthetic_data(
    n_samples_total=2000,
    resample_size=200
)

# Plot
synth.plot_distributions(show_rarity=True, compare_resampled=True)
synth.plot_augmented_vs_original()

# Metrics
metrics = synth.generate_metrics()
print(metrics)



fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the histogram for "Original" data
ax2 = ax1.twinx()  # Create a second y-axis
ax2.hist(df['Rings'], bins=30, alpha=0.3, color='gray', label="Original", density=True)
ax2.set_ylabel("Histogram Frequency of Original Distribution")

# Plot KDE plots for other datasets
sns.kdeplot(df['Rings'], label="Original", ax=ax1)
sns.kdeplot(augmented_df['Rings'], label="Augmented", ax=ax1)

# Labels
ax1.set_xlabel("Rings")
ax1.set_ylabel("Density")
ax1.legend()

# Show the plot
plt.show()
