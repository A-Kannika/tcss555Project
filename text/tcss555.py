import csv
import sys
import argparse
import os
from text import main as text_main


# run program with the arguments using argparse
# https://docs.python.org/3/howto/argparse.html
# Don't check the directory for now.

def main():
    text_main()
    # parser = argparse.ArgumentParser(description='Check input and output directory paths.')

    # parser.add_argument('-i', '--input_dir', required=True, type=str,
    #                     help='Path to the input directory')
    # parser.add_argument('-o', '--output_dir', required=True, type=str,
    #                     help='Path to the output directory')

    # args = parser.parse_args()
    # save_to_XML_file(args.input_dir, args.output_dir)


def is_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    return path


# Generate the output with parsing arguments from the command
def save_to_XML_file(input_dir, output_dir):
    # Open the profile in the public-test-data
    # Command arguments
    # tcss555 -i /data/public-test-data/ -o ~/output/

    input_path = os.path.join(input_dir, "profile", "profile.csv")
    with open(input_path, mode="r") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            userid = row['userid']

            # XML String to be saved to file
            xml_string = f"""<user
    id=\"{userid}\"
age_group=\"xx-24\"
gender=\"female\"
extrovert=\"3.486857895\"
neurotic=\"2.732424211\"
agreeable=\"3.583904211\"
conscientious=\"3.445616842\"
open=\"3.908690526\"
/>"""

            # File path to save the XML file
            file_path = os.path.join(output_dir, f"{userid}.xml")
            # Open file in write mode
            with open(file_path, "w", encoding="utf-8") as out_file:
                # Write the XML string to the file
                out_file.write(xml_string)
            # File saved successfully
            # print(f"{userid}.xml saved to", file_path)


if __name__ == "__main__":
    main()


