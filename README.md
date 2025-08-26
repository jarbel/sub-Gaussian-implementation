python -m venv venv
venv\Scripts\activate

pip install -U pip setuptools wheel
pip install black flake8 isort mypy pytest pytest-cov sphinx

pip install pyinstaller
pyinstaller --noconsole --add-data "assets;assets" core/GUI_proxy.py

numpy 
scipy
matplotlib
tk
pillow
pandas 