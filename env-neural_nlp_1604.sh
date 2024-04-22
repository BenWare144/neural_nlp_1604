##################
### Install miniconda
##################
# requirements:
sudo apt install git
# install guide: https://docs.conda.io/projects/miniconda/en/latest/
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
# to initialize:
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# sudo apt install curl -y
# cd /tmp
# https://docs.conda.io/projects/miniconda/en/latest/
# curl https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
# bash Miniconda* -u
conda update conda
# conda install -n base -c conda-forge jupyterlab jupyterlab_widgets

# To upgrade miniconda, download the lastest and use "-u" option to upgrade

# source ~/.bashrc;export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

##################
### manage git versions
##################
sudo apt install git
mkdir -p ~/code_orig/github_orig
git -C /home/ben/code_orig/github_orig clone https://github.com/mschrimpf/neural-nlp
git -C /home/ben/code_orig/github_orig clone https://github.com/brain-score/brainio_base
git -C /home/ben/code_orig/github_orig clone https://github.com/brain-score/brainio_collection
git -C /home/ben/code_orig/github_orig clone https://github.com/brain-score/brain-score
git -C /home/ben/code_orig/github_orig clone https://github.com/brain-score/result_caching

git -C /home/ben/code_orig/github_orig/neural-nlp         checkout -f 1a896f2 # fixes bug introduced through ^^
git -C /home/ben/code_orig/github_orig/brainio_base       checkout -f 6e56cfd
git -C /home/ben/code_orig/github_orig/brainio_collection checkout -f 0adbea7
git -C /home/ben/code_orig/github_orig/brain-score        checkout -f 61d7a7b # on Feb 12, 2020 (Little cleanup) 
git -C /home/ben/code_orig/github_orig/result_caching     checkout -f 94cb58c
###    manually fix their dependencies
# old. ???? **comment out all "brainio", "result_caching", and "text" dependencies** in all 5 packages primary "setup.py" files
# old. ???? also comment out "jupyter", if desired

##################
### Install neural_nlp_1604
##################

conda env create -f /home/ben/neural_nlp_1604.yaml
# conda remove -n neural_nlp_1code_orig604 --all
if [ $CONDA_DEFAULT_ENV != "neural_nlp_1604" ]; then conda activate neural_nlp_1604; fi
pip install dask[array]==2.1.0
pip install git+https://github.com/mschrimpf/lm_1b.git@1ff7382 # - lm-1b==0.0.1
pip install git+https://github.com/mschrimpf/OpenNMT-py.git@f339063 # - opennmt-py==0.2
pip install git+https://github.com/mschrimpf/skip-thoughts.git@c8a3cd5 # - skip-thoughts==0.0.1
pip install git+https://github.com/nltk/nltk_contrib.git@c9da2c2 # - nltk-contrib==3.4.1 (from here: https://github.com/nltk/nltk_contrib/tree/python3)
# text:         pip install git+https://github.com/pytorch/text.git@0.3.1 # or just pip install torchtext==0.3.1 
# pip install --force-reinstall  numpy==1.20.1 # becuse ?????
# pip install numpy==1.19 # Gets rid of stupid errors

pip install -e ~/code_orig/github_orig/brainio_base --no-deps
pip install -e ~/code_orig/github_orig/brainio_collection --no-deps
pip install -e ~/code_orig/github_orig/result_caching --no-deps
pip install -e ~/code_orig/github_orig/brain-score --no-deps
pip install -e ~/code_orig/github_orig/neural-nlp --no-deps

pip install pickle5
##################
### Install neural_nlp_1604_cpu
##################

conda env create --force -f /media/ben/705c0330-41a6-49b9-995d-d15adde369a7/home/ben/code/conda_envs/neural_nlp_1604_cpu.yaml
if [ $CONDA_DEFAULT_ENV != "neural_nlp_1604_cpu" ]; then conda activate neural_nlp_1604_cpu; fi
# conda remove -n neural_nlp_1604 --all
# conda install pytorch=1.5.1=py3.7_cpu_0 cpuonly=2.0=0 pytorch-mutex=1.0=cpu -c pytorch
pip install dask[array]==2.1.0
pip install git+https://github.com/mschrimpf/lm_1b.git@1ff7382 --no-deps # - lm-1b==0.0.1
pip install git+https://github.com/mschrimpf/OpenNMT-py.git@f339063 --no-deps # - opennmt-py==0.2
pip install git+https://github.com/mschrimpf/skip-thoughts.git@c8a3cd5 --no-deps # - skip-thoughts==0.0.1
pip install git+https://github.com/nltk/nltk_contrib.git@c9da2c2 --no-deps # - nltk-contrib==3.4.1 (from here: https://github.com/nltk/nltk_contrib/tree/python3)
# text:         pip install git+https://github.com/pytorch/text.git@0.3.1 # or just pip install torchtext==0.3.1 
# pip install --force-reinstall  numpy==1.20.1 # becuse ?????
# pip install numpy==1.19 # Gets rid of stupid errors

pip install -e ~/code_orig/github_orig/brainio_base --no-deps
pip install -e ~/code_orig/github_orig/brainio_collection --no-deps
pip install -e ~/code_orig/github_orig/result_caching --no-deps
pip install -e ~/code_orig/github_orig/brain-score --no-deps
pip install -e ~/code_orig/github_orig/neural-nlp --no-deps

pip install pickle5

##################
### Run neural_nlp_1604
##################

alias ntop="
nvidia-settings -a [GPU-0]/GPUFanControlState=1 >/dev/null 
nvidia-settings -a [FAN-0]/GPUTargetFanSpeed=100
&& watch -n 2 nvidia-smi"

cd ~/code_orig/github_orig/neural-nlp/
if [ $CONDA_DEFAULT_ENV != "neural_nlp_1604" ]; then conda activate neural_nlp_1604; fi
# if [ $CONDA_DEFAULT_ENV != "neural_nlp_1604_cpu" ]; then conda activate neural_nlp_1604_cpu; fi
python neural_nlp run --model gpt2 --benchmark Blank2014fROI-encoding --log_level DEBUG
rm -rf /home/ben/.result_caching/neural_nlp.models.wrapper.core.ActivationsExtractorHelper._from_sentences_stored ; python neural_nlp run --model gpt2 --benchmark Blank2014fROI-encoding --log_level DEBUG
# conda env create --force -f /media/ben/705c0330-41a6-49b9-995d-d15adde369a7/home/ben/code_orig/test.yaml

















##################
### Test
##################
# conda env create --force -f /media/ben/705c0330-41a6-49b9-995d-d15adde369a7/home/ben/code_orig/test.yaml




##################
### Other
##################

for x in Blank2014fROI.nc Blank2014fROI-encoding-ceiling.nc Blank2014fROI-encoding-ceiling-bootstrapped_params.nc Blank2014fROI-encoding-ceiling-raw.nc Blank2014fROI-encoding-ceiling-raw-bootstrapped_params.nc Blank2014fROI-encoding-ceiling-raw-endpoint_x.nc Blank2014fROI-encoding-ceiling-raw-raw.nc Blank2014fROI-stimulus_set.csv ; do
diff "~/.neural_nlp/$x" "~/.neural_nlp (copy)/$x"
done


diff "~/.neural_nlp/Blank2014fROI.nc" "~/.neural_nlp (copy)/Blank2014fROI.nc"
Blank2014fROI.nc 
Blank2014fROI-encoding-ceiling.nc 
Blank2014fROI-encoding-ceiling-bootstrapped_params.nc 
Blank2014fROI-encoding-ceiling-raw.nc 
Blank2014fROI-encoding-ceiling-raw-bootstrapped_params.nc 
Blank2014fROI-encoding-ceiling-raw-endpoint_x.nc 
Blank2014fROI-encoding-ceiling-raw-raw.nc 
Blank2014fROI-stimulus_set.csv













