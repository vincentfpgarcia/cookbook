# Pyenv

## Description

Pyenv lets you easily switch between multiple versions of Python.

## Installation

Follow the installation instructions on the [pyenv](https://github.com/pyenv/pyenv)  repository.

Don't forget to setup your shell environment (e.g. bashrc and zshrc).

## How to use Pyenv

Pyenv is a command line tool used in a Terminal.

To list the available versions of Python, use the command:

```
$ pyenv install --list
```

To install for instance Python 3.9.16 (if available), use the command:

```
$ pyenv install 3.9.16
```

At this point, you are not yet using the Python version you've installed. If you use the following command, you'll be able to see what is the version of Python used:

```
$ pyenv versions
* system
  3.9.16 (set by /Users/foo/.pyenv/version)
```

The `*` means that the Python version used is the one from the system. To change that, use the command:

```
$ pyenv global 3.9.16
```

If you now use the `pyenv versions` command, you'll see that the Python version used is 3.9.16.

It is possible to specify a different version of Python to use only in a given folder. To do so, simply use:

```
$ pyenv local 3.9.16
```
