set -ex

CONFIG=$1
python -m diffusion.train --config "$CONFIG"