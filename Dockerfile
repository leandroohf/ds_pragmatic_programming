# This installs  python 3.6 as well as jupyter and other python packages
FROM jupyter/tensorflow-notebook

RUN conda install -y -c conda-forge jupyter_contrib_nbextensions
RUN conda install -y -c r rpy2
RUN conda install -y -c conda-forge matplotlib
RUN conda install -y -c plotly plotly

# Setup jupyter to avoid tokens/passwords
USER root
RUN apt-get update \
    #&& apt-get -y install time \
    && apt-get install -y --no-install-recommends curl graphviz \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /home/jovyan/data
RUN mkdir -p /home/jovyan/images

RUN chown jovyan:users /home/jovyan/data
RUN chown jovyan:users /home/jovyan/images

ADD images /home/jovyan/images
ADD data  /home/jovyan/data
COPY README.md /home/jovyan
COPY *.ipynb /home/jovyan/


USER jovyan

# install jupyter extensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable --py widgetsnbextension


# XXX:extension collapse sections not working
#RUN jupyter nbextensions_configurator enable --user
RUN  jupyter nbextension enable collapsible_headings/main
RUN  jupyter nbextension enable codefolding/main

