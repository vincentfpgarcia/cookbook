# Matplotlib

## Backend

As described in the [official documentation](https://matplotlib.org/faq/usage_faq.html#what-is-a-backend), _the “frontend” is the user facing code, i.e., the plotting code, whereas the “backend” does all the hard work behind-the-scenes to make the figure._ Matplotlib supports different backends. Please see the full documentation for a complete presentation.

I first present how to display which backend is used by your Python. I then present 2 common methods to specify the backend to use.


### Display the backend used

To display the backend used in Python, one can use the following Python code:

```python
import matplotlib
print('Backend used: %s' % matplotlib.rcParams['backend'])
```

On my computer, this gives me the following output:

```
Backend used: agg
```


### Change the backend in the code

One can change the backend directly in the Python code:

```python
import matplotlib

# Display current backend
print('Backend before change: %s' % matplotlib.rcParams['backend'])

# Change the backend
matplotlib.use('MacOSX')

# Verify current backend
print('Backend after change:  %s' % matplotlib.rcParams['backend'])
```

On my computer, this gives me the following output:

```
Backend before change: agg
Backend after change:  MacOSX
```

### Change the backend using `matplotlibrc`

One can customize Matplotlib by specifying some information in the `matplotlibrc`:

- Edit the file `matplotlibrc` using a standard text editor. On my Macbook, this file is located at `/Users/username/.matplotlib/matplotlibrc`. If the file does not exist, simply create it.
- Add the following line (Replace `TkAgg` by the backend of your choice), save and exit:

```
backend: TkAgg
```

If you now display the backend used in Python using

```python
import matplotlib
print('Backend used: %s' % matplotlib.rcParams['backend'])
```

you should get the following output:

```
Backend used: TkAgg
```