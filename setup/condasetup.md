conda create --name torch_cuda python=3.11 matplotlib numpy scikit-learn pandas jupyterlab lightning pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda create --name torch_cuda python=3.11 matplotlib numpy scikit-learn pandas jupyterlab pytorch-lightning pytorch torchaudio cudatoolkit=11.8 -c pytorch


### conda create command

``` bash
conda create --name torch python=3.11 matplotlib numpy scikit-learn pandas lightning pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
```
