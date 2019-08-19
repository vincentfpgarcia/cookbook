# Python and Anaconda


## Installation

- Go to the [Anaconda download page](https://www.anaconda.com/distribution)
- Download the Python 3 graphical installer
- Install Anaconda (follow the steps)


## Conda

Conda is a package, dependency and environment management for any language. It is installed by Anaconda, but let's see if the `conda` command is available. Open a terminal and try to execute the command:

```
% conda
```

If conda command is not available, add conda to your `PATH` by executing the following command in a terminal:

```
% export PATH=/anaconda3/bin/:$PATH
```

The conda command should be available now in this terminal and as long as you don't close it.

You can also add the previous command in your `.bashrc` (or `.zshrc`) file if you want the `conda ` to be available every time a terminal is opened.


## Virtual environments

### Creation

To create a virtual environment nammed `myenv` (any name will do):

```
% conda create -n myenv
```

To specify the version of Python we want in the environment:

```
% conda create -n myenv python=3.7
```

### Activation / Deactivation

Activate the environment:

```
> source activate myenv
```

Deactivate the environment:

```
> source deactivate
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

## Install packages

To install a package:

```
% conda install scipy
```

We can also target a specific environment:

```
% conda install -n myenv scipy
```

And we can specify the version of the package we want to use:

```
% conda install -n myenv scipy=0.15.0
```

Note that packages can also be installed directly when the environment is created.


## Jupyter notebook and virtual environment

To use a virtual environment inside Jupyter Notebook, first we need to install `ipykernel:`

```
% conda install -n myenv ipykernel
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