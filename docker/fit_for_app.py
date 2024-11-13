import sys
from jax import config

config.update("jax_enable_x64", True)

from tsadar.runner import run_for_app

if __name__ == "__main__":
    run_for_app(sys.argv[1])
