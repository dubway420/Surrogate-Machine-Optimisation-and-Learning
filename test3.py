from os import listdir
from machine_learning.utils import is_in

search_name = "CNN_64_32_16"

def experiment_number_finder(search_name, directory):
    """ Find the experiment number of the experiment with the given name in the given directory 
    
    search_name (string): this is the name of the experiment you are trying to find

    directory (string): this is the folder that contains the files to search through. This is usually 
    the parent file of the trial

    Returns: experiment number (integer), this is the experiment number of the experiment you are 
    trying to find. Taken from the file name i.e. if the search_name was found in experiment3.py, this
    method will return 3.

    """



    # Get the list of files in the directory
    path = directory + "experiment_input_files"
    files = listdir(path)

    # Find the experiment number
    # Do this by iterating through the files in the experiment_input_files directory
    for file in files:

        # If the file has "experiment" in the title, then it is a experimentX.py files
        if is_in(file, "experiment"): 

            # Get just the number of the experiment i.e. remove the string "experiment" and the 
            # extension ".py"
            experiment_number = file.replace("experiment", "").replace(".py", "")

            # Read all lines of the experimentX.py file until the experiment name label is found
            with open("experiment_input_files/" + file, "r") as f:
                lines = f.readlines()

                for line in lines:

                    # This is the identifying line
                    if is_in(line, "experiment_name = "):

                        # Get just the name of the experiment and stop searching through this 
                        # particular file 
                        experiment_name = line.split("=")[1].strip()
                        break

            # If the file carries the correct experiment name, return the experiment number            
            if is_in(experiment_name, search_name):
                return experiment_number     
            



            
           


                



# from experiment_input_files.experiment1 import Features

# print(Features)





