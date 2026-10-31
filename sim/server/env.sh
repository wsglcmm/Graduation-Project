# 用法：source /mnt/2disk/vicelead/env.sh
export WS=/mnt/2disk/vicelead
export VL=$WS                      # 10/26 版 env.sh 使用的名稱，保留相容
export HOME=$WS/home
export TMPDIR=$WS/tmp
export TMUX_TMPDIR=$WS/tmp
export XDG_CACHE_HOME=$HOME/.cache
export XDG_CONFIG_HOME=$HOME/.config
export XDG_DATA_HOME=$HOME/.local/share
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export MAMBA_ROOT_PREFIX=$WS/micromamba
export PATH=$WS/bin:$PATH
export PYTHONPATH=$WS/code/lib${PYTHONPATH:+:$PYTHONPATH}
[ -x $WS/bin/micromamba ] && eval "$($WS/bin/micromamba shell hook -s bash)"
micromamba activate px4 2>/dev/null
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
cd $WS
export GZ_IP=127.0.0.1
