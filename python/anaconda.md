# Python and Anaconda


## Installation

- Go to the [Anaconda download page](https://www.anaconda.com/distribution)
- Download the Python 3 graphical installer
- Install Anaconda (follow the steps)


## Conda

Conda is a package, dependency and environment management for any language. It is installed by Anaconda, but let's see if the `conda` command is available. Open a Terminal and try to execute the command:

```
% conda
```

If conda command is not available, it's probably because the initialization has not been done. Anaconda recommends to use the `conda init` command.

On my computer, Anaconda 3 has been installed in the `/Users/vincent/opt/anaconda3` folder. Note that this folder might be different on your system. To initialize conda, use the following command line in a Terminal. Note that I'm using zsh as shell. This initialization also works with other shells.

```
% /Users/vincent/opt/anaconda3/bin/conda init zsh
```

This command will modify the file `.zshrc` to add the conda command.

When restarting the Terminal, one can see that the `(base)` conda environment is used by default. If you do not want to start automatically in a conda environment, use the following command before the `conda init`:

```
% conda config --set auto_activate_base false
```

Now, when starting a Terminal, you should not be in a conda environment.

The second option to add conda to your environment is to do it manually. Again, this is not recommended by Anaconda. To do so, simply add the conda folder to your `PATH` using the following command in a Terminal:

```
% export PATH=/Users/vincent/opt/anaconda3/bin/:$PATH
```

The conda command should be available now in this Terminal and as long as you don't close it. If you want the `conda` command to be available every time a Terminal is opened, add the previous command at the end of your `.zshrc` file (Remember that  I'm using zsh. If you are using bash, modify the `.bashrc` file).

See [FAQ](https://docs.anaconda.com/anaconda/user-guide/faq/) for more information about Anaconda.


## Virtual environments

### Creation

To create a virtual environment nammed `myenv` (any name will do):

```
% conda create --name myenv
```

To specify the version of Python we want in the environment:

```
% conda create --name myenv python=3.7
```

### Activation / Deactivation

Activate the environment:

```
% conda activate myenv
```

Deactivate the environment:

```
% conda deactivate
```


### Management

To see all the available virtual environments in conda:

```
% conda env list
```

or 

```
% conda info --envs
```

To remove an environment:

```
% conda remove --name myenv --all
```

### Verification of Python

We want to make sure that the Python being used in the environment is the correct one. To do so, first activate the `myenv` environment. Then, enter Python using the `python command`. This will give something like that:

```
(myenv) ➜ ~ python
Python 3.7.6 (default, Jan 8 2020, 13:42:34)
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

We can see that the Python version used is correct and is somehow related to Anaconda. Now, we can verify the Python interpreter using the following Python code:

```
>>> import sys
>>> print(sys.executable)
/Users/vgarcia/opt/anaconda3/envs/myenv/bin/python
```

And now, we have the confirmation that the interpreter used in `myenv` is the one stored in the `myenv` folder. So it's all good!



## Install packages

To install a package:

```
% conda install scipy
```

We can also target a specific environment:

```
% conda install --name myenv scipy
```

And we can specify the version of the package we want to use:

```
% conda install --name myenv scipy=0.15.0
```

Note that packages can also be installed directly when the environment is created.

To display the packages installed:

```
% conda list
```


## Jupyter notebook and virtual environment

To use a virtual environment inside Jupyter Notebook, first we need to install `ipykernel:`

```
% conda install --name myenv ipykernel
```

Then, we create a IPython kernel for Jupyter:

```
% source activate myenv
% python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

In Jupyter notebook now, select the `Python (myenv)`  kernel in the menu `Kernel` > `Change kernel` to execute Python code inside the `myenv` environment.


## Sublime Text and virtual environment

The content of this section was inspired by the [official documentation of Anaconda](https://docs.anaconda.com/anaconda/user-guide/tasks/integration/sublime/). Please read Anaconda's documentation for a complete presentation of how to integrate Conda virtual environments within Sublime Text. You'll find bellow a procedure that works for me:

1. Open Sublime Text
2. Install the Conda package:
	- Open the `Command palette` menu using the menu `Tools > Command Palette...`
	- Access the `Package Control : Install Package` section
	- Search for the `Conda` package and install it
3. Change the current Build System to `Conda` in the menu `Tools > Build System`
4. Select the virtual environment you want to use:
	- Open the `Command palette` menu using the menu `Tools > Command Palette...`
	- Access the `Conda: Activate Environment` section
	- Choose the environment to use in the list
5. If the previous step does not show your environment(s), it may be because the environment directory needs to be set:
	- Go to menu `Sublime Text > Preferences > Package Settings > Conda > Settings - User`
	- Define the variable `environment_directory ` to point to the location of the Anaconda environments location. In my case, Conda environments are store in `/anaconda3/envs/`. So in my case, the `Settings - User` looks like this:

```
{
	"environment_directory": "/anaconda3/envs/"
}
```