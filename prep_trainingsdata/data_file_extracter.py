import os
import json
import read_data


def find_files_containing_task(directory_path, target_text, output_json_path="output.json"):
    """
    Searches for a specific text within all text files in a directory
    and writes the names of the files containing the text to a JSON file.

    Args:
        directory_path (str): The path to the directory containing the text files.
        target_text List(str): The text to search for within the files.
        output_json_path (str, optional): The path to the output JSON file.
            Defaults to "output.json".
    """
    files_containing_text = []
    files_not_containing_text = []
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".h5"):  # Consider only .txt files, adjust if needed
            try:
                # Use codecs.open to handle various encodings
                data = read_data.load_h5_file(directory_path + filename)
                file_dict = {}
                for segment in data["segments_info"].values():
                    if segment["text"].decode("utf-8") == "Insert round peg 4.":
                        file_dict["file"] = filename
                        file_dict["start"] = segment["start"]
                        file_dict["end"] = segment["end"]
                        file_dict["action"] = segment["text"].decode("utf-8")
                        files_containing_text.append(file_dict)
                if file_dict == {}:
                    files_not_containing_text.append(filename)
            except Exception as e:
                print(f"Error reading file '{filename}': {e}")
                # Consider logging the error or handling it as needed

    # Write the results to a JSON file
    try:
        with open(output_json_path, "w") as json_file:
            json.dump(files_containing_text, json_file, indent=4)
        print(f"Files containing the text '{target_text}' were written to '{output_json_path}'")
    except Exception as e:
        print(f"Error writing to JSON file '{output_json_path}': {e}")
        # Consider logging the error

if __name__ == "__main__":
    # Get the directory path and target text from the user
    directory_path = "/home/cgreiml/Documents/extract_data_for_training/vis_lib/data/"
    target_text = [b"Insert round peg 4.", b"Insert round peg 3."]
    output_json_path = "traj_aut.json"

    # Call the function to find and save the files
    find_files_containing_task(directory_path, target_text, output_json_path)