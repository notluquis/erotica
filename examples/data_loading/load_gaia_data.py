#!/usr/bin/env python3
"""Data loading example using COSMIC.

This example demonstrates how to load and preprocess Gaia data
using COSMIC's DataLoader and DataPreprocessor classes.
"""
from pathlib import Path
import cosmic

def main():
    """Demonstrate data loading workflow."""
    print("📊 COSMIC Data Loading Example")
    print("="*35)
    
    # Check for sample data
    data_path = Path("../../data/data.ecsv")
    if not data_path.exists():
        print(f"\n⚠️  Sample data not found at {data_path}")
        print("This example requires sample data to run.")
        print("Please ensure you have the COSMIC sample data available.")
        return
    
    # 1. Initialize DataLoader
    print(f"\n1. Loading data from {data_path}...")
    loader = cosmic.DataLoader(str(data_path))
    
    # 2. Check available photometric systems
    print("\n2. Checking available photometric systems...")
    systems = loader.check_available_photometric_systems()
    for system, info in systems.items():
        print(f"   {system}: {info['available_columns']} columns available")
    
    # 3. Load data with multiple photometric systems
    print("\n3. Loading Gaia + 2MASS data...")
    data = loader.load_data(
        systems=['Gaia', 'TMASS'],
        include_distances=['photogeometric'],
        normalize_names=True
    )
    print(f"   Loaded {len(data)} sources with {len(data.colnames)} columns")
    
    # 4. Display data info
    print("\n4. Data Information:")
    print(f"   Columns: {', '.join(data.colnames[:10])}...")
    print(f"   RA range: {data['ra'].min():.2f} to {data['ra'].max():.2f} deg")
    print(f"   Dec range: {data['dec'].min():.2f} to {data['dec'].max():.2f} deg")
    
    # 5. Count valid sources
    print("\n5. Valid source counts:")
    counts = loader.count_valid_sources()
    for category, count in counts.items():
        print(f"   {category}: {count} sources")
    
    # 6. Initialize preprocessor
    print("\n6. Setting up data preprocessing...")
    preprocessor = cosmic.DataPreprocessor(data)
    
    # 7. Run preprocessing steps
    print("\n7. Running preprocessing pipeline...")
    print("   - Filling missing values...")
    preprocessor.fill_missing_values()
    
    print("   - Renaming columns for consistency...")
    preprocessor.rename_columns()
    
    print("   - Applying zero-point corrections...")
    try:
        preprocessor.apply_zero_point_correction()
    except Exception as e:
        print(f"     Warning: {e}")
    
    print("   - Correcting proper motions...")
    preprocessor.correct_proper_motion()
    
    print("   - Adding photometric errors...")
    preprocessor.add_photometric_errors()
    
    print("   - Dropping invalid sources...")
    n_before = len(data)
    preprocessor.drop_invalid_sources(['pmra', 'pmdec', 'parallax'])
    n_after = len(preprocessor.data)
    print(f"     Kept {n_after}/{n_before} sources ({100*n_after/n_before:.1f}%)")
    
    # 8. Split by data quality
    print("\n8. Splitting data by quality...")
    good_data, bad_data, stats = preprocessor.filter_data(fidelity_threshold=0.5)
    
    print("   Quality split results:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    print("\n✅ Data loading and preprocessing completed!")
    print(f"\nReady for clustering with {len(good_data)} high-quality sources")
    print("\nNext steps:")
    print("- Use good_data for clustering analysis")
    print("- See examples/basic_clustering/ for clustering workflow")
    print("- Check examples/visualization/ for data exploration plots")

if __name__ == "__main__":
    main()