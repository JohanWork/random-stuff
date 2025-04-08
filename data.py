# Create a summary DataFrame using pandas
import pandas as pd

cluster_summary = []
for i in range(n_clusters):
    cluster_examples = dataset.filter(lambda x: x['cluster'] == i)
    
    # Get unique contexts in this cluster
    unique_contexts = len(set(cluster_examples['positive']))
    
    summary = {
        'Cluster': i,
        'Number of Examples': len(cluster_examples),
        'Unique Contexts': unique_contexts,
        'Sample Question': cluster_examples[0]['anchor'][:100],
        'Sample Context': cluster_examples[0]['positive'][:100]
    }
    cluster_summary.append(summary)

summary_df = pd.DataFrame(cluster_summary)
print("\nCluster Summary:")
print(summary_df)
