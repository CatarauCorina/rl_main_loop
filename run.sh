sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
pip install virtualenv
chmod +x /content/drive/MyDrive/rl_main_loop/rl_main_loop/bin/pip
chmod +x /content/drive/MyDrive/rl_main_loop/rl_main_loop/bin/python
source /content/drive/MyDrive/rl_main_loop/rl_main_loop/bin/activate || virtualenv rl_main_loop --python=python3.8
pip install -r requirements_main_lopp.txt
pip -V;
python --version
pip install seaborn
pip install scipy==1.9.3
pip freeze
python baseline_models/trainer.py
