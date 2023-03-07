# Black

## Description

From [Black's website](https://github.com/psf/black):  Black is the uncompromising Python code formatter. By using it, you agree to cede control over minutiae of hand-formatting. In return, Black gives you speed, determinism, and freedom from pycodestyle nagging about formatting. You will save time and mental energy for more important matters.

## How to install (using Poetry)

If you use poetry, installing black is very simple:

```
$ poetry add black
```

## How to use (using Poetry)

Black is a command line tool that can be used on one file or on an entire directory. I assume here you are already in the virtual environment:

```
$ black src/foo.py
$ black .
```

Black's command accepts lots of parameters, all explained using the following command:

```
$ black --help
```

For instance, the following command will print in color the differences before and after formatting, but it does not apply the formatting:

```
$ black src/foo.py --diff --color
```

## Customization

Black's behaviour can be customized by addind a section in the `pyproject.toml` file. Here is an example:

```
[tool.black]
line-length = 100
```