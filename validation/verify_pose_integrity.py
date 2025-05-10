import numpy as np
import os
import argparse


def is_rotation_matrix(R):
    should_be_identity = np.dot(R.T, R)
    I = np.identity(3, dtype=R.dtype)
    error = np.linalg.norm(I - should_be_identity)
    return error < 1e-3


def check_pose_file(pose_file):
    pose = np.load(pose_file)
    if pose.shape != (3, 4):
        return False, "Invalid shape"

    R = pose[:, :3]
    t = pose[:, 3]
    if not is_rotation_matrix(R):
        return False, "Non-orthogonal rotation matrix"

    return True, "Valid"


def main(directory):
    pose_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    passed = 0

    for file in pose_files:
        valid, msg = check_pose_file(os.path.join(directory, file))
        if valid:
            passed += 1
        else:
            print(f"[FAIL] {file}: {msg}")

    print(f"\nChecked {len(pose_files)} files.")
    print(f"Passed: {passed}/{len(pose_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .npy pose files")
    args = parser.parse_args()
    main(args.dir)
