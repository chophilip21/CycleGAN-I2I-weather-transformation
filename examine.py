import os
import json
from pathlib import Path

def examine_dataset(dataset_path):
    """
    Examine the dataset structure and check for:
    1. Matching number of files in frames and segmentations_mseg folders for each sequence
    2. Empty folders
    3. Missing sequence folders
    """
    issues = []
    stats = {
        'total_sequences': 0,
        'matching_sequences': 0,
        'mismatching_sequences': 0,
        'missing_sequences': 0,
        'empty_sequences': 0
    }

    # Convert to Path object for easier handling
    dataset_path = Path(dataset_path)

    # Check if path exists and is a directory
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise ValueError(f"Path {dataset_path} does not exist or is not a directory")

    # Get all sequence folders in the frames directory
    frames_dir = dataset_path / 'frames'
    seg_dir = dataset_path / 'segmentations_mseg'

    if not frames_dir.exists():
        issues.append({
            'type': 'missing_folder',
            'path': str(frames_dir),
            'details': 'frames directory is missing'
        })
        return

    if not seg_dir.exists():
        issues.append({
            'type': 'missing_folder',
            'path': str(seg_dir),
            'details': 'segmentations_mseg directory is missing'
        })
        return

    # Get all sequence folders
    frame_sequences = {d.name for d in frames_dir.iterdir() if d.is_dir()}
    seg_sequences = {d.name for d in seg_dir.iterdir() if d.is_dir()}

    # Check for missing sequences
    missing_in_frames = seg_sequences - frame_sequences
    missing_in_seg = frame_sequences - seg_sequences

    if missing_in_frames:
        stats['missing_sequences'] += len(missing_in_frames)
        issues.append({
            'type': 'missing_sequences',
            'details': {
                'missing_in_frames': list(missing_in_frames)
            }
        })

    if missing_in_seg:
        stats['missing_sequences'] += len(missing_in_seg)
        issues.append({
            'type': 'missing_sequences',
            'details': {
                'missing_in_segmentations': list(missing_in_seg)
            }
        })

    # Check common sequences
    common_sequences = frame_sequences.intersection(seg_sequences)
    stats['total_sequences'] = len(common_sequences)

    for seq in common_sequences:
        frames_path = frames_dir / seq
        seg_path = seg_dir / seq

        # Get file counts
        frames_files = [f for f in frames_path.iterdir() if f.is_file()]
        seg_files = [f for f in seg_path.iterdir() if f.is_file()]

        # Check for empty folders
        if not frames_files or not seg_files:
            stats['empty_sequences'] += 1
            issues.append({
                'type': 'empty_sequence',
                'path': str(seq),
                'details': {
                    'frames_count': len(frames_files),
                    'segmentations_count': len(seg_files)
                }
            })

        # Check for matching file counts
        if len(frames_files) != len(seg_files):
            stats['mismatching_sequences'] += 1
            issues.append({
                'type': 'mismatching_files',
                'path': str(seq),
                'details': {
                    'frames_count': len(frames_files),
                    'segmentations_count': len(seg_files)
                }
            })
        else:
            stats['matching_sequences'] += 1

    # Print summary
    print("\nDataset Examination Report:")
    print("-" * 50)
    print(f"Total sequences examined: {stats['total_sequences']}")
    print(f"Sequences with matching files: {stats['matching_sequences']}")
    print(f"Sequences with mismatching files: {stats['mismatching_sequences']}")
    print(f"Missing sequences: {stats['missing_sequences']}")
    print(f"Empty sequences: {stats['empty_sequences']}")
    print("\nDetailed Issues:")
    print("-" * 50)

    # Group and print issues by type
    issue_types = set(issue['type'] for issue in issues)
    for issue_type in sorted(issue_types):
        print(f"\n{issue_type.upper()}")
        print("=" * 50)
        for issue in [i for i in issues if i['type'] == issue_type]:
            if issue_type == 'missing_sequences':
                print("\nMissing in frames:", json.dumps(issue['details'].get('missing_in_frames', []), indent=2))
                print("Missing in segmentations:", json.dumps(issue['details'].get('missing_in_segmentations', []), indent=2))
            elif issue_type == 'mismatching_files':
                print(f"\nSequence: {issue['path']}")
                print("Details:", json.dumps(issue['details'], indent=2))
            elif issue_type == 'empty_sequence':
                print(f"\nSequence: {issue['path']}")
                print("Details:", json.dumps(issue['details'], indent=2))
            else:
                print(f"\nPath: {issue['path']}")
                if 'details' in issue:
                    print("Details:", json.dumps(issue['details'], indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Examine dataset structure')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset root directory')
    args = parser.parse_args()
    
    try:
        examine_dataset(args.dataset_path)
    except Exception as e:
        print(f"Error: {str(e)}")
