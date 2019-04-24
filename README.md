# Random-Forest

The following usage instruction refers to the 'single-tree' version.
This version is a tag accessible by:
	
		> git checkout single-tree

Usage:

        > main.py dataset_filename [delimiter_char]

**Args:**

* dataset_filename:
	
	>path to the dataset csv file.

* delimiter_char: optional

	>character that delimites the dataset.
    >The default value is ';'

**Examples:**

    	> main.py data.csv ;
    	> main.py relative/path/data.csv
    	> main.py /absolute/path/data.csv



## Why Pipfile

Pipfile is used by the [pipenv](https://realpython.com/pipenv-guide/) module.  
with this, we can be sure we are all running the same python and dependencies version.  

### Pipenv Installation
pipenv can be installed with [pip](https://docs.python.org/3/installing/index.html), as any other dependency.

	> pip install pipenv

#### Creating a virtual environment from the Pipfile

	> cd Pipfile_directory
	> pipenv install

#### Running the virtual environment

	> cd Pipfile_directory
	> pipenv shell
	(Random-Forest)> *inside the env*

#### Installing new dependencies to the virtual environment

while not running the virtual env:

	> cd Pipfile_directory
	> pipenv install package_name

It will include the dependency in the Pipfile, so these changes have to be committed.  
Running pip install inside the virtual env will install the package just to the env, but wont be includede in the Pipfile.


## Linux

On linux, you may have the following error when running the GUI
	
	ModuleNotFoundError: No module named 'Tkinter'

In this case, you'll have to install the package python3-tk in order to run the gui version.

	sudo apt-get install python3-tk

---