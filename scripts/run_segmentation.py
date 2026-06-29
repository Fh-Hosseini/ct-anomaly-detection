"""
Run TotalSegmentator on a single CT volume.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/hpc/iwi5/iwi5437h/ct-anomaly-detection")
sys.path.insert(0, str(PROJECT_ROOT))

from src.ct_anomaly.data.segmentation import segment_one_volume


def main():
    volume_name = sys.argv[1]
    input_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    print(f"Processing: {volume_name}")

    segment_one_volume(
        volume_path=input_path,
        mask_dir=output_dir,
        device="gpu",
    )

    print(f"Done: {volume_name}")


if __name__ == "__main__":
    main()