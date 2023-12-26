import shutil
import subprocess

def copy_and_backup(source):
    # Create a backup in the same location with '_backup' appended to the filename
    backup_destination = source.replace('.csv', '_backup.csv')

    # Copy the source file to the backup destination
    shutil.copy(source, backup_destination)

    print(f"Backup created for {source} at {backup_destination}")

def run_data_scripts():
    # Run data cleaning script
    subprocess.run(["python", "./data_cleaning.py"])

    # Run data switching script
    subprocess.run(["python", "./data_switching.py"])

    # Run data augmentation script
    subprocess.run(["python", "./data_augmentation.py"])

if __name__ == "__main__":
    # Copy and backup dev.csv
    copy_and_backup("../../data/dev.csv")

    # Copy and backup train.csv
    copy_and_backup("../../data/train.csv")

    # Run data scripts in order
    run_data_scripts()