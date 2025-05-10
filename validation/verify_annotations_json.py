import json
import argparse
import numpy as np

def check_quaternion(q):
    norm = np.linalg.norm(q)
    return abs(norm - 1.0) < 1e-3

def main(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    total = len(annotations)
    passed = 0

    for ann in annotations:
        q = ann.get("rotation", [])
        if check_quaternion(q):
            passed += 1
        else:
            print(f"[FAIL] Image ID {ann.get('image_id')} has invalid quaternion: {q}")

    print(f"\nChecked {total} annotations.")
    print(f"Passed: {passed}/{total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to posecnn_annotations.json")
    args = parser.parse_args()
    main(args.json)
