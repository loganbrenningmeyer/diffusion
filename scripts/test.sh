set -ex

CONFIG=$1
python -m diffusion.test --config "$CONFIG"