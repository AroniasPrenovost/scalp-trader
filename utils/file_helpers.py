# boilerplate
import os
from dotenv import load_dotenv
load_dotenv()
# end boilerplate

def count_files_in_directory(directory_path):
    """
    Returns the number of files in the specified directory.

    :param directory_path: Path to the directory
    :return: Number of files in the directory
    """
    try:
        # List all entries in the directory
        entries = os.listdir(directory_path)

        # Filter out directories, only count files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]

        return len(files)
    except Exception as e:
        print(f"Error counting files in directory {directory_path}: {e}")
        return None
