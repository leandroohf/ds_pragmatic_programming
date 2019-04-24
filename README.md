# Data Science Pragmatic Programming

  Snippet codes for common used routines in data science projects.

  [python](ds_pragmatic_programming_python.ipynb)


## Automated system

### Spark notebook

You need to run inside a docker container that has all packages and spark
running


https://hub.docker.com/r/ucsddse230/cse255-dse230



```sh
# Install images
docker pull ucsddse230/cse255-dse230

# Run container
docker run --name ds_pragmatic -it -p 8890:8888 -v /media/leandroohf/sdb1/leandro/ds_pragmatic_programming:/home/ucsddse230/ ucsddse230/cse255-dse230 /bin/bash

# Run jupyter inside container
jupyter notebook


# If you need to ssh to container
docker exec -it ds_pragmatic /bin/bash

```

1. http://localhost:8890/tree
2. Copy and paste token to login in the notebook


You do not have to, but in case you want to build the images

```sh
mkdir -p builddir
cd builddir

# !?
curl asdfa

docker build -t ucsddse230/cse255-dse230 .

# Run for testing
docker run -it -p 8888:8888 ucsddse230/cse255-dse230 jupyter notebook 
```

 Un compress wetaher data
 
 ```sh
 tar -zxvf data/Weather/NY.tgz -C data/Weather
 ```
