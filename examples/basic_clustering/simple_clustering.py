#!/usr/bin/env python3
"""Simple clustering example using COSMIC.

This example demonstrates the basic workflow for clustering astronomical data
using COSMIC's simplified API.
"""
import numpy as np
from astropy.table import QTable
import astropy.units as u

# Import COSMIC modules
import cosmic

def create_sample_data(n_sources=1000):
    """Create sample astronomical data for demonstration."""
    # Generate synthetic star cluster data
    np.random.seed(42)
    
    # Cluster center in proper motion space
    pmra_center, pmdec_center = 5.0, -2.0  # mas/yr
    parallax_center = 2.0  # mas (500 pc distance)
    
    # Generate cluster members (70% of data)
    n_cluster = int(0.7 * n_sources)
    pmra_cluster = np.random.normal(pmra_center, 0.5, n_cluster)
    pmdec_cluster = np.random.normal(pmdec_center, 0.5, n_cluster)
    parallax_cluster = np.random.normal(parallax_center, 0.2, n_cluster)
    
    # Generate field stars (30% of data)
    n_field = n_sources - n_cluster
    pmra_field = np.random.normal(0, 3.0, n_field)
    pmdec_field = np.random.normal(0, 3.0, n_field)
    parallax_field = np.random.normal(1.0, 0.8, n_field)
    
    # Combine data
    pmra = np.concatenate([pmra_cluster, pmra_field])
    pmdec = np.concatenate([pmdec_cluster, pmdec_field])
    parallax = np.concatenate([parallax_cluster, parallax_field])
    
    # Generate other required columns
    source_id = np.arange(n_sources)
    ra = np.random.uniform(280, 285, n_sources)
    dec = np.random.uniform(-25, -20, n_sources)
    gmag = np.random.uniform(12, 18, n_sources)
    
    # Create QTable
    data = QTable({
        'source_id': source_id,
        'ra': ra * u.deg,
        'dec': dec * u.deg,
        'pmra': pmra * u.mas/u.yr,
        'pmdec': pmdec * u.mas/u.yr,
        'parallax': parallax * u.mas,
        'Gmag': gmag * u.mag,
    })
    
    return data

def main():
    """Run basic clustering example."""
    print("🌌 COSMIC Basic Clustering Example")
    print("="*40)
    
    # 1. Create sample data
    print("\n1. Creating sample astronomical data...")
    data = create_sample_data(n_sources=1000)
    print(f"   Generated {len(data)} sources")
    
    # 2. Set up clustering
    print("\n2. Setting up HDBSCAN clustering...")
    clusterer = cosmic.Clustering(data)
    
    # 3. Run clustering
    print("\n3. Running clustering analysis...")
    clusterer.search(
        columns=['pmra', 'pmdec', 'parallax'],
        param_grid={'min_cluster_size': [10, 20, 30]},
    )
    
    # 4. Display results
    print("\n4. Clustering Results:")
    print(f"   Best parameters: {clusterer.get_best_params()}")
    print(f"   Best score: {clusterer.best_score_:.4f}")
    
    # 5. Get cluster summary
    print("\n5. Cluster Summary:")
    summary = clusterer.get_cluster_summary()
    print(summary[['cluster', 'count', 'persistence']].to_string(index=False))
    
    # 6. Display statistics
    print("\n6. Clustering Statistics:")
    clusterer.clustering_statistics()
    
    print("\n✅ Basic clustering example completed!")
    print("\nNext steps:")
    print("- Try examples/visualization/ for plotting results")
    print("- See examples/advanced_clustering/ for parameter optimization")
    print("- Check examples/ngc6383/ for real data analysis")

if __name__ == "__main__":
    main()