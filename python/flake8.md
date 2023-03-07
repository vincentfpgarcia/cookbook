# Flake8

## Description

Flake8 is a wrapper around the tools PyFlakes, pycodestyle and Ned Batchelder's McCabe script.


## How to install (using Poetry)

If you use poetry, installing flake8 is very simple:

```
$ poetry add flake8
```

## How to use (using Poetry)

Flake8 is a command line tool. I assume here you are already in the virtual environment:

```
$ flake8
```

Flake8's command accepts lots of parameters, all explained using the following command:

```
$ flake8 --help
```

## Customization

Flake8's behaviour can be customized through a `.flake8` file. Here is an example:

```
[flake8]
max-line-length = 100
count = true
exclude = .git,.venv
```