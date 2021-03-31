# Docker


## Images

### Create images

First, go to the folder than contains your Dockerfile.

Build an image from a Dockerfile:

```
$ docker build .
```

Build an image from a Dockerfile and give the image a name / tag:

```
$ docker build --tag my-name:my-tag .
```

### List and remove images

List images:

```
$ docker image ls
```

List all images including intermediate images:

```
$ docker image ls -a
```

List images but only show their ID:

```
$ docker image ls -q
```

Remove on image given its ID, for instance fe4f5ad8a20c:

```
$ docker image rm fe4f5ad8a20c
```

Remove all images:

```
$ docker image rm $(docker image ls -q)
```

## Containers
