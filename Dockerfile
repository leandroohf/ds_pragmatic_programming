# This installs  python 3.6 as well as jupyter and other python packages
FROM jupyter/pyspark-notebook

# Install nbgrader and pyspark
RUN conda install -y -c anaconda numpy
RUN conda install -y -c conda-forge nbgrader
RUN conda install -y beautifulsoup4
RUN conda install -y Flask
RUN conda install -y -c conda-forge jupyter_contrib_nbextensions
RUN conda install -y -c conda-forge ipyleaflet
RUN conda install -y -c anaconda basemap
RUN conda install -y -c conda-forge findspark
RUN conda install -y -c anaconda pydot
RUN conda install -y -c conda-forge matplotlib
RUN conda install -y -c conda-forge xgboost
RUN conda install -y -c damianavila82 rise
RUN conda install -y -c conda-forge pydotplus
RUN conda install -y -c conda-forge tensorflow
RUN pip install jupyter-tensorboard

# Setup jupyter to avoid tokens/passwords
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl graphviz \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/jovyan/.jupyter
RUN chown jovyan:users /home/jovyan/.jupyter
WORKDIR /home/jovyan/.jupyter
RUN /opt/conda/bin/jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '*'" >> jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = u''" >> jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> jupyter_notebook_config.py
RUN chown jovyan:users jupyter_notebook_config.py
RUN usermod -l ucsddse230 jovyan
RUN usermod -d /home/ucsddse230 -m ucsddse230
ENV HOME "/home/ucsddse230"
WORKDIR /home/ucsddse230/work
USER ucsddse230
