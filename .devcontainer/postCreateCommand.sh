#!/usr/bin/env bash

# set -ex

echo "******************************************"
echo "Welcome to the PyTorch Docker from Jearde!"
echo "******************************************"

tmux set-option mouse on
tmux set -g allow-passthrough on

if [ -z "$GIT_NAME" ]
then
    echo "GIT_NAME is not defined. Not setting git config."
else 
    echo "Setting git config..."
    git config --global user.email $GIT_EMAIL
    git config --global user.name $GIT_NAME
    git config --global core.editor "code-insiders --wait"
    git config --global pull.rebase true
    git config --global push.default current
    git config --global --add --bool push.autoSetupRemote true
    git config --global --add safe.directory /workspaces/*
fi

echo "Container specifications:"
echo ""

echo "User: $(whoami)"
echo "OS: $(lsb_release -a | grep Description:)"
echo "Python: $(which python)"
echo "Python Version: $(python -V)"
nvidia-smi
nvidia-smi -L
python -c '
import torch
import torchvision
import torchaudio
import lightning as L
print(f"PyTorch Version: {torch.__version__}")
print(f"Lightning Version: {L.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Supported CUDA Architectures: {torch.__version__, torch.cuda.get_arch_list()}")
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"\tGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"\t\tDevice Properties: {torch.cuda.get_device_properties(i)}")
'


CC=$(python -c '
import torch, sys
try:
    p = torch.cuda.get_device_properties(0)
    print(f"{p.major}{p.minor}")
except Exception:
    sys.exit(1)
')

echo "Compute capability is $CC"

if [ -z "$CC" ] || [ "$CC" -le 70 ]; then
    echo "Installing legacy torch package for ${CC} <= 7.0"
    uv pip install -r /workspace/requirements.txt;
    uv pip uninstall torch torchvision torchaudio;
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126;
else
    echo "Package not installed because compute capability is $CC (> 7.0)"
fi

echo "*******************************************************************************"


# nohup bash -c 'tensorboard --logdir ${LOG_DIR}/tensorboard --port 6007 --host 0.0.0.0 &'
# echo "Tensorboard started on http://localhost:6007"

# nohup bash -c 'mlflow ui --backend-store-uri ${LOG_DIR}/mlflow --port 8080 --host 0.0.0.0 &'
# echo "Tensorboard started on http://localhost:6007"

# optuna-dashboard sqlite:///log/optuna.sqlite3 --host  
# optuna-dashboard sqlite:////mnt/nfs/log/optuna/optuna.sqlite3 --host 0.0.0.0 --port 8085 &

# code-insiders tunnel --accept-server-license-terms --name="${TUNNEL_NAME}"