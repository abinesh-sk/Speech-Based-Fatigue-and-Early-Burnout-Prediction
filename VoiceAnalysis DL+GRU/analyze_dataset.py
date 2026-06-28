"""
Script to analyze dataset statistics from manifest files.
Generates a comprehensive report of dataset distribution.
"""

import json
from pathlib import Path
from collections import Counter

def analyze_manifest(manifest_path):
    """Analyze a single manifest file."""
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    polarity_counts = Counter(item['polarity'] for item in data)
    dataset_counts = Counter(item['dataset'] for item in data)
    
    return {
        'total': total,
        'polarity': dict(polarity_counts),
        'datasets': dict(dataset_counts)
    }

def print_statistics():
    """Print comprehensive dataset statistics."""
    base_path = Path(__file__).parent / 'data' / 'processed' / 'features'
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    print("="*80)
    print("DATASET STATISTICS REPORT")
    print("="*80)
    
    # Collect statistics for each split
    for split in splits:
        manifest_path = base_path / split / 'manifest.json'
        if manifest_path.exists():
            stats[split] = analyze_manifest(manifest_path)
        else:
            print(f"\nWarning: {manifest_path} not found!")
            stats[split] = None
    
    # Calculate totals
    total_samples = sum(s['total'] for s in stats.values() if s)
    
    print(f"\n{'OVERALL SUMMARY'}")
    print("-"*80)
    print(f"{'Total Samples:':<30} {total_samples:>10,}")
    print(f"{'Training Samples:':<30} {stats['train']['total']:>10,} ({stats['train']['total']/total_samples*100:>6.2f}%)")
    print(f"{'Validation Samples:':<30} {stats['val']['total']:>10,} ({stats['val']['total']/total_samples*100:>6.2f}%)")
    print(f"{'Test Samples:':<30} {stats['test']['total']:>10,} ({stats['test']['total']/total_samples*100:>6.2f}%)")
    
    # Per-split breakdown
    print("\n" + "="*80)
    print("SPLIT-WISE BREAKDOWN")
    print("="*80)
    
    for split in splits:
        if not stats[split]:
            continue
            
        print(f"\n{split.upper()} SET")
        print("-"*80)
        print(f"{'Total:':<30} {stats[split]['total']:>10,}")
        print(f"\n{'Class Distribution:'}")
        print(f"  {'Class':<20} {'Count':<15} {'Percentage'}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        
        for polarity in ['negative', 'neutral', 'positive']:
            count = stats[split]['polarity'].get(polarity, 0)
            pct = count / stats[split]['total'] * 100
            print(f"  {polarity:<20} {count:<15,} {pct:>6.2f}%")
    
    # Overall class distribution
    print("\n" + "="*80)
    print("OVERALL CLASS DISTRIBUTION (All Splits Combined)")
    print("="*80)
    
    total_by_class = {
        'negative': sum(stats[s]['polarity'].get('negative', 0) for s in splits if stats[s]),
        'neutral': sum(stats[s]['polarity'].get('neutral', 0) for s in splits if stats[s]),
        'positive': sum(stats[s]['polarity'].get('positive', 0) for s in splits if stats[s])
    }
    
    print(f"\n{'Class':<20} {'Count':<15} {'Percentage'}")
    print(f"{'-'*20} {'-'*15} {'-'*15}")
    for polarity, count in total_by_class.items():
        pct = count / total_samples * 100
        print(f"{polarity:<20} {count:<15,} {pct:>6.2f}%")
    
    # Dataset sources
    print("\n" + "="*80)
    print("DATASET SOURCES")
    print("="*80)
    
    all_datasets = set()
    for split in splits:
        if stats[split]:
            all_datasets.update(stats[split]['datasets'].keys())
    
    print(f"\n{'Dataset':<20} {'Train':<15} {'Val':<15} {'Test':<15} {'Total'}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for dataset in sorted(all_datasets):
        train_count = stats['train']['datasets'].get(dataset, 0) if stats['train'] else 0
        val_count = stats['val']['datasets'].get(dataset, 0) if stats['val'] else 0
        test_count = stats['test']['datasets'].get(dataset, 0) if stats['test'] else 0
        total = train_count + val_count + test_count
        print(f"{dataset:<20} {train_count:<15,} {val_count:<15,} {test_count:<15,} {total:,}")
    
    # Detailed class distribution per split
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ACROSS SPLITS")
    print("="*80)
    
    print(f"\n{'Class':<20} {'Train':<15} {'Val':<15} {'Test':<15} {'Total'}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for polarity in ['negative', 'neutral', 'positive']:
        train_count = stats['train']['polarity'].get(polarity, 0) if stats['train'] else 0
        val_count = stats['val']['polarity'].get(polarity, 0) if stats['val'] else 0
        test_count = stats['test']['polarity'].get(polarity, 0) if stats['test'] else 0
        total = train_count + val_count + test_count
        print(f"{polarity:<20} {train_count:<15,} {val_count:<15,} {test_count:<15,} {total:,}")
    
    print("\n" + "="*80)
    print("REPORT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    print_statistics()
