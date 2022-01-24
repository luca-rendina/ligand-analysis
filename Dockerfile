FROM continuumio/miniconda3
MAINTAINER Greg Landrum <greg.landrum@gmail.com>

ENV PATH /opt/conda/bin:$PATH
ENV LANG C

# install the RDKit:
RUN conda config --add channels  https://conda.anaconda.org/rdkit

# note including jupyter in this brings in rather a lot of extra stuff
RUN conda install -y nomkl rdkit pandas cairo jupyter

RUN apt-get update && apt-get install -y libxrender-dev
RUN mkdir /my_data

#INSTALLING PACKAGE
RUN conda install -y scikit-learn matplotlib pydotplus python-graphviz
RUN conda install -y -c oddt oddt
RUN conda install -y -c conda-forge pypdb biopandas py3dmol
RUN conda install -y pip
RUN alias pip3='opt/conda/bin/pip3'
RUN pip3 install pygtop
RUN conda install -y -c mcs07 pubchempy
RUN conda install -y -c openeye openeye-toolkits
RUN conda install -y -c conda-forge nglview

RUN apt-get update -y
RUN apt-get install -y autodock-vina

RUN conda install -y -c anaconda biopython 
RUN conda install -y -c conda-forge keras
RUN pip3 install keras-ncp
RUN conda install -y -c conda-forge xgboost
RUN conda install -y -c conda-forge openbabel 
RUN conda install -y -c conda-forge h5py
RUN conda install -y -c anaconda requests
RUN conda install -y -c conda-forge tensorflow
RUN conda install -y -c conda-forge keras
RUN pip3 install tensorflow-addons[tensorflow]
RUN conda install -y -c anaconda seaborn

EXPOSE 8888

CMD jupyter notebook --notebook-dir=/my_data --ip='0.0.0.0' --allow-root --NotebookApp.token=''
