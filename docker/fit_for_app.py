import sys
from jax import config
import jax

config.update("jax_enable_x64", True)

from tsadar.runner import run_for_app

if __name__ == "__main__":
    # print jax devices
    print("jax devices: ", jax.devices())

    run_for_app(sys.argv[1])
