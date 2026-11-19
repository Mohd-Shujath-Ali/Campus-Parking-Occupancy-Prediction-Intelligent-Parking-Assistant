import os
import shutil
import subprocess
from pathlib import Path

repo_dir = Path(r"D:\Academic\Semester 5\ML\Campus-Parking-Occupancy-Prediction-Intelligent-Parking-Assistant")
compressed_rf = Path(r"D:\Academic\Semester 5\ML\Supervised\artifacts\pipelines\rf_forecaster.joblib")

def run(cmd, check=True):
    print(f">> {cmd}")
    res = subprocess.run(cmd, cwd=repo_dir, shell=True, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print("STDERR:", res.stderr.strip())
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{res.stderr}")
    return res

print(f"=== Rebuilding Git History without >100MB File ===")
print(f"Compressed rf_forecaster size: {compressed_rf.stat().st_size / (1024*1024):.2f} MB")

# 1. Ensure clean working tree
run("git status")

# 2. Store original commit hashes
commits = [
    "f76571a", # Multi-model benchmarking
    "286c687", # Documentation
    "7268f3a", # Notebooks
    "4a0b3ef", # Edge cases
    "f6678bc", # Decoupled Render architecture
    "062d51c", # WebGPU YOLO
]

# 3. Create clean branch from origin/main
run("git checkout -B clean-main origin/main")

# 4. Cherry-pick first commit without committing
run(f"git cherry-pick --no-commit {commits[0]}")

# 5. Overwrite the 102MB file with the 29.4MB compressed version
target_rf = repo_dir / "artifacts" / "pipelines" / "rf_forecaster.joblib"
shutil.copy2(compressed_rf, target_rf)
print(f"Replaced target rf_forecaster with compressed: {target_rf.stat().st_size / (1024*1024):.2f} MB")
run(f"git add artifacts/pipelines/rf_forecaster.joblib")
run(f"git commit -C {commits[0]}")

# 6. Cherry-pick remaining commits
for c in commits[1:]:
    run(f"git cherry-pick {c}")

# 7. Move clean-main to main
run("git checkout -B main clean-main")

# 8. Clean up temporary refs
run("git branch -D clean-main", check=False)

print("\n=== Verifying Git History ===")
run("git log origin/main..main --oneline")

print("\n=== Verifying Max File Size in Rebuilt History ===")
check_script = """
for obj in $(git rev-list origin/main..main); do
    git ls-tree -r -l $obj | awk '{print $4, $5}' | sort -n -k1 | tail -n 5
done
"""
res = subprocess.run(["git", "rev-list", "origin/main..main"], cwd=repo_dir, capture_output=True, text=True)
revs = res.stdout.strip().split()
max_size = 0
max_file = ""
for rev in revs:
    tree_out = subprocess.run(["git", "ls-tree", "-r", "-l", rev], cwd=repo_dir, capture_output=True, text=True).stdout
    for line in tree_out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].isdigit():
            size = int(parts[3])
            name = parts[4]
            if size > max_size:
                max_size = size
                max_file = name

print(f"Largest file in unpushed history: {max_file} ({max_size / (1024*1024):.2f} MB)")
if max_size < 100 * 1024 * 1024:
    print("SUCCESS: All files are strictly below GitHub's 100MB limit!")
else:
    print("WARNING: A file still exceeds 100MB!")
