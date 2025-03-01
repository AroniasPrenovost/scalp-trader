# boilerplate
import os
from dotenv import load_dotenv
load_dotenv()
# end boilerplate
import time

def count_files_in_directory(directory):
    """
    Returns the number of files in the specified directory.

    :param directory: Path to the directory
    :return: Number of files in the directory
    """
    try:
        # List all entries in the directory
        entries = os.listdir(directory)

        # Filter out directories, only count files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory, entry))]

        return len(files)
    except Exception as e:
        print(f"Error counting files in directory {directory}: {e}")
        return None


# Function to delete files older than a specified number of hours
def delete_files_older_than_x_hours(directory, hours):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    cutoff_time = time.time() - (hours * 3600)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            if file_creation_time < cutoff_time:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")


# Function to check if the most recent file is older than 30 minutes
def is_most_recent_file_older_than_x_minutes(directory, minutes):
    if not os.path.exists(directory):
        return True

    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        return True

    most_recent_file = max(files, key=os.path.getctime)
    file_creation_time = os.path.getctime(most_recent_file)
    return (time.time() - file_creation_time) > (minutes * 60)
