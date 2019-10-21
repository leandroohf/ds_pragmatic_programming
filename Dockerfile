# This installs  python 3.6 as well as jupyter and other python packages
FROM jupyter/tensorflow-notebook

RUN conda install -y -c conda-forge jupyter_contrib_nbextensions
RUN  conda install -y -c r rpy2
RUN conda install -y -c conda-forge matplotlib

# Setup jupyter to avoid tokens/passwords
USER root
RUN apt-get update \
    && apt-get -y install time \
    && apt-get install -y --no-install-recommends curl graphviz \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*


## TODO: with root user did not worked
## TODO: try again with /home/jovyan/ and set permission to data forlder
# RUN mkdir -p /home/jovyan/.jupyter
# RUN chown jovyan:users /home/jovyan/.jupyter
# WORKDIR /home/jovyan/.jupyter
# RUN /opt/conda/bin/jupyter notebook --generate-config
# RUN echo "c.NotebookApp.ip = '*'" >> jupyter_notebook_config.py
# RUN echo "c.NotebookApp.open_browser = False" >> jupyter_notebook_config.py
# RUN echo "c.NotebookApp.token = u''" >> jupyter_notebook_config.py
# RUN echo "c.NotebookApp.allow_root = True" >> jupyter_notebook_config.py
# RUN chown jovyan:users jupyter_notebook_config.py
# RUN usermod -l ucsddse230 jovyan
# RUN usermod -d /home/ucsddse230 -m ucsddse230
# ENV HOME "/home/ucsddse230"
# WORKDIR /home/ucsddse230/work
# USER ucsddse230


RUN mkdir -p /home/jovyan/data
RUN mkdir -p /home/jovyan/images

RUN chown jovyan:users /home/jovyan/data
RUN chown jovyan:users /home/jovyan/images

ADD images /home/jovyan/images
ADD data  /home/jovyan/data
COPY README.md /home/jovyan
COPY *.ipynb /home/jovyan/

RUN whoami
RUN pwd
RUN ls 
