# isort

## Description

From [isort's website](https://pycqa.github.io/isort/):  isort is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

## How to install (using Poetry)

If you use poetry, installing black is very simple:

```
$ poetry add isort
```

## How to use (using Poetry)

isort is a command line tool that can be used on one file or on an entire directory. I assume here you are already in the virtual environment:

```
$ isort src/foo.py
$ isort .
```

isort's command accepts lots of parameters, all explained using the following command:

```
$ isort --help
```

For instance, the following command will print in color the differences before and after formatting, but it does not apply the formatting:

```
$ isort src/foo.py --diff
```

## Customization

isort's behaviour can be customized by addind a section in the `pyproject.toml` file. Here is an example:

```
[tool.isort]
profile = "black"
```