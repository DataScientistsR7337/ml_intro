# A Very quick Introduction to ML using Jupyterlab and pyTorch

## Prerequisities

Install miniconda (small) (or Anaconda: big and not really necessary):

<https://docs.conda.io/projects/conda/en/latest/user-guide/install/>

If on Windows: Please allow it to add itself yo our path environment variable.


Once miniconda is installed, bring up the miniconda shell (if on windows), or ordinary shell (Linux) and create a new Python environment for our experiments. 

NOTE: Everything you do in these environments (installing packages and tools) are local that exactly that environment. No risk of contaminating the system version of Python.

```
conda create -n mnistlab -c conda-forge --override-channels python=3.8
```
Then start using that environment by

```
conda activate mnistlab
```

Now, we want quite a few tools for our ML lab.
Let's install them, by specifying them on the commandline. It's more packages than we will use in this session, but I rather give a complete list of the most useful ML tools than just a subset.
```
conda install -c conda-forge jupyterlab  matplotlib numpy scikit-learn seaborn tensorboard 
conda install -c conda-forge -c pytorch -c pytorch torchvision
pip install hiddenlayer
```

The installation is about 200MB + 880MB. It takes time (several minutes) so please be patient.

If the conda installation of pyTorch is extremely slow, you may try pip using the selector at this site:
<https://pytorch.org/>

In my case, it was 
```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```
But depending on GPU hardware (if it exists) your setup may be different.


Once it is complete, you can start the tutorial

## Your notebook template

We will be using JupyterLab, which is an improved version of Jupyter Notebooks.
Launch it with
```
jupyter lab 
```

This will open up your browser and redirect it to the JupyterLab page. 
Upload the mnistlab.pynb file to Jupyterlab open it. The tutorial continues in the notebook.