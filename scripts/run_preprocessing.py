"""
Run preprocessing pipeline for CT-RATE-AD.
"""

import sys
from pathlib import Path
from src.ct_anomaly.data.preprocessing import preprocess_volume


def main():
    volume_name = sys.argv[1]
    input_path = Path(sys.argv[2])
    mask_dir = Path(sys.argv[3])
    output_path = Path(sys.argv[4])

    print(f"Processing: {volume_name}")

    preprocess_volume(
        volume_path=input_path,
        mask_dir=mask_dir,
        output_path=output_path,
    )

    print(f"Done: {volume_name}")


if __name__ == "__main__":
    main()