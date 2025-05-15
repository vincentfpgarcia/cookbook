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

Remove an image given its ID, for instance fe4f5ad8a20c:

```
$ docker image rm fe4f5ad8a20c
```

Remove all images:

```
$ docker image rm $(docker image ls -q)
```

## Containers

### Create containers

Create a container:

```
$ docker run my-image
```

Redirect the container port to your local port:

```
$ docker run -p 8080:8080 my-image
```

Define an environement variable:

```
$ docker run -e PORT=8080 my-image
```

### List, stop and remove containers

List runnning containers:

```
$ docker ps
```

List all containers:

```
$ docker ps -a
```

List all containers but only show their IDs:

```
$ docker ps -aq
```

Stop a container given its ID, for instance b025bfad7503:

```
$ docker kill b025bfad7503
```

Stop all containers:

```
$ docker kill $(docker ps -aq)
```

Remove a container given its ID, for instance b025bfad7503:

```
$ docker rm b025bfad7503
```

Remove all containers:

```
$ docker rm $(docker ps -aq)
```


## Useful Tools

Launches an interactive Bash shell inside a new container based on the "agent" image:

```
docker run -it my-image bash
```

Get a bash running in a container:

```
$ docker exec -it b025bfad7503 /bin/bash
```

Get the container ID from the container name:

```
$ docker ps -aqf ancestor=my-image
```

Get the stats of a given container:

```
$ docker stats b025bfad7503
```

Get the memory used by a container:

```
$ docker stats b025bfad7503 --no-stream --format "{{{{.MemUsage}}}}
```

Clean-up, remove unused data:

```
$ docker system prune -af
```