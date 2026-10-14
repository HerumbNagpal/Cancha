import os

print("\nSTEP 1 - Download Latest Data")
os.system("python scripts/download_data.py")

print("\nSTEP 2 - Merge Datasets")
os.system("python scripts/update_data.py")

print("\nSTEP 3 - Feature Engineering")
os.system("python scripts/feature_engineering.py")

print("\nSTEP 4 - Train Model")
os.system("python scripts/train_model.py")

print("\nPipeline Complete.")