sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get install -y -qq python3.10-full python3.10-dev

python3.10 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -U wheel
pip install -r requirements.txt
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html