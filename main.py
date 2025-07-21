import os
import subprocess
import sys


def run_script(script_name):
    print(f"\n{'=' * 50}")
    print(f"Running {script_name}...")
    print(f"{'=' * 50}")

    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {script_name} failed!")
        return False


def check_file_exists(filename):
    if os.path.exists(filename):
        print(f"✅ Found {filename}")
        return True
    else:
        print(f"❌ Missing {filename}")
        return False


print("🚀 Book Recommendation System Training Pipeline")
print("🚀 Training all models locally")

print(f"\n{'=' * 50}")
print("CHECKING FOR DATA...")
print(f"{'=' * 50}")

if not check_file_exists("book_dataset.csv"):
    print("\n❌ No book dataset found!")
    print("Please run: python collect_book_data.py first")
    sys.exit(1)

print(f"\n{'=' * 50}")
print("STARTING TRAINING PIPELINE...")
print(f"{'=' * 50}")

scripts_to_run = [
    "train_baseline_models.py",
    "train_vae_model.py"
]

all_success = True

for script in scripts_to_run:
    if not run_script(script):
        all_success = False
        break

print(f"\n{'=' * 60}")
if all_success:
    print("🎉 ALL TRAINING SCRIPTS COMPLETED!")
    print("🎉 SUCCESS! All models trained successfully!")
    print("📚 Your book recommendation system is ready!")
    print("\nNext steps:")
    print("  1. Run: streamlit run streamlit_app.py")
    print("  2. Or test individual models with the separate scripts")
else:
    print("❌ TRAINING PIPELINE FAILED!")
    print("Please check the error messages above")

print(f"{'=' * 60}")