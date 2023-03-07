# Poetry

## Description

Poetry is a tool for dependency management and packaging in Python.

## Installation

Follow the installation instructions available on the [poetry](https://python-poetry.org) page.

On macOS, install poetry using the command:

```
$ curl -sSL https://install.python-poetry.org | python3 -
```

Then add the poetry to your PATH. In my case, I needed to add the following line to my `.zshrc` file as I was using using ZSH:

```
export PATH="$HOME/.local/bin:$PATH"
```

## Virtual environments in project

Poetry creates virtual environments. Optionally, these environments can be create within the considered project. To do so, use the following command:

```
$ poetry config virtualenvs.in-project
```

When the project is installed (see bellow), a `.venv` is available in the project root directory.


## Use poetry with an existing project

At the root of the project, there should be a `pyproject.toml` file and potentially a `poetry.lock` file.

To create the virtual environment, use the command:

```
$ poetry install
```

Then, execute your Python code (contained for instance in a `src/main.py` file) within the virtual environement using the command:

```
$ poetry run python src/main.py
```

Alternatively, you can first ask poetry to spawn a shell within the virtual environment. Then, you just need to call your python code without using the `poetry run` command:

```
$ poetry shell
$ python src/main.py
```

## Use poetry to create a new project

Create a new empty project using:

```
$ poetry new poetry-demo
```

This will create a directory that looks like this:

```
poetry-demo
├── pyproject.toml
├── README.md
├── poetry_demo
│   └── __init__.py
└── tests
    └── __init__.py
```

Using the `--src` option will place the `poetry-demo` directory within a `src` directory:

```
$ poetry new --src my-project
```

Then, you can add your code within the appropriate directory and execute it for instance using:

```
$ poetry run python poetry_demo/main.py
```

## Add dependencies


### General case

Poetry lets you add dependencies to your project using for instance:

```
$ poetry add numpy
```

### Groups

Poetry provides a way to organize your dependencies by groups. For instance, you might have dependencies that are only needed to test your project or to build the documentation. To add a dependency to a group, use:

```
$ poetry add some-package --group dev
```

By default, when using the `poetry install` command, Poetry will install all groups (except for the optional ones). It is possible to exclude groups from being installed:

```
$ poetry install --without dev
```

### Optional

A dependency group can be declared as optional using:

```
$ poetry add some-package --group my-group --optional
```

By default, when using the `poetry install` command, poetry will not install optional groups. It is possible to include optional groups using:

```
$ poetry install --with my-group
```

### Lock

The list of dependencies can be locked and placed within a `poetry.lock` using:

```
$ poetry lock
```

When such a file exists, poetry will install dependencies from the `poetry.lock` file instead of from the `pyproject.toml` file.