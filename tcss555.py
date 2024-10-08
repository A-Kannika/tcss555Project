import csv
import sys
import argparse


# run program with the arguments using argparse
# https://docs.python.org/3/howto/argparse.html
def run_program():
    parser = argparse.ArgumentParser()



# Generate the output and save to the XML file
def save_to_XML_file():
    # Open the profile in the public-test-data
    # The file path need to change to the parsing arguments
    with open("data/public-test-data/profile/profile-test.csv", "r") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            userid = row['userid']
            print(f"hello {userid} ")
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

            # FIle path to save the XML file
            file_path = f"{userid}.xml"
            # Open file in write mode
            with open(file_path, "w", encoding="utf-8") as out_file:
                # Write the XML string to the file
                out_file.write(xml_string)
            # File saved successfully
            print(f"{userid}.xml saved to", file_path)

if __name__ == "__main__":
    # test
    save_to_XML_file()