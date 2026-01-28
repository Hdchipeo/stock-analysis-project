import sys
import os

# Add src to python path to facilitate imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import collect_data
import analyze_data
import descriptive_stats
import preprocess_data
import eda_analysis
import modeling

def main():
    print("===========================================")
    print("   STOCK ANALYSIS PIPELINE                 ")
    print("===========================================")

    # Phase 1: Data Collection
    print("\n--- [Phase 1] Data Collection ---")
    collect_data.collect_stock_data(ticker="FPT.VN")
    analyze_data.analyze_and_describe_variables()

    # Phase 2: Descriptive Statistics
    print("\n--- [Phase 2] Descriptive Statistics ---")
    descriptive_stats.calculate_descriptive_stats()

    # Phase 3: Data Preprocessing
    print("\n--- [Phase 3] Data Preprocessing ---")
    # This generates preprocessed_data.csv, train_data.csv, test_data.csv
    # and outliers.png in results/figures (via updated script)
    preprocess_data.preprocess_stock_data()

    # Phase 4: EDA & Visualization
    print("\n--- [Phase 4] EDA & Visualization ---")
    # This generates png files in results/figures
    eda_analysis.run_eda_analysis()

    # Phase 5: Modeling
    print("\n--- [Phase 5] Modeling ---")
    # This generates model_comparison.png, feature_importance.png in results/figures
    modeling.run_modeling()

    print("\n===========================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY         ")
    print("===========================================")

if __name__ == "__main__":
    main()
