from deep_traffic_generation.core import cli_main
from deep_traffic_generation.core.datasets import TrafficDataset
from deep_traffic_generation.fcvae import FCVAE

if __name__ == "__main__":
    cli_main(FCVAE, TrafficDataset, "linear", seed=42)