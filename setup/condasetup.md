conda create --name torch_cuda python=3.11 matplotlib numpy scikit-learn pandas jupyterlab lightning pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda create --name torch_cuda python=3.11 matplotlib numpy scikit-learn pandas jupyterlab lightning pytorch torchaudio cudatoolkit=11.8 -c pytorch


### conda create command with cuda toolkit

``` bash
conda create --name torch python=3.11 matplotlib numpy scikit-learn pandas lightning pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
```

### conda create command with cuda toolkit

``` bash
conda create --name torch python=3.11 matplotlib numpy scikit-learn pandas lightning pytorch torchvision torchaudio -c pytorch -c nvidia 
```

### Version von Basil

``` bash
mamba create --name torch_insect python=3.11 matplotlib numpy scikit-learn pandas lightning pytorch torchvision torchaudio pytorch jupyterlab tensorboard -c pytorch -c nvidia
pip install soundfile
```

### version for ubuntu

``` bash
conda create --name torch_cuda python=3.11 matplotlib numpy scikit-learn pandas jupyterlab lightning pytorch torchvision torchaudio pytorch-cuda=11.8 cudatoolkit=11.8 tensorboard -c pytorch -c nvidia

pip install soundfile

