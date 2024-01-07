import os


def rename_files_in_folder(folder_path, prefix):
    i = 1  # Start numbering from 1
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Check if it's a JPG file
            new_name = f"{prefix}{i}.jpg"
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_name)

            # Check if the new file name already exists
            while os.path.exists(new_file):
                i += 1  # Increment the number if file exists
                new_name = f"{prefix}{i}.jpg"
                new_file = os.path.join(folder_path, new_name)

            os.rename(old_file, new_file)
            i += 1  # Increment for the next file


folder_path = r"D:\Repo\\Python\faacedetct\images\face"
rename_files_in_folder(folder_path, "face")
