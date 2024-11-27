from src import path_to_config
from src.script.metric_collector import MetricsVisualizer
from src.utils.config_parser import ConfigParser
from src.utils.custom_logging import setup_logging
from env import Env

log = setup_logging()
env = Env()


def train_plotter():
    config = ConfigParser.parse(path_to_config())
    collector_config = config.get('MetricsVisualizer', {})
    collector = MetricsVisualizer(path_to_metrics=env.__getattr__("METRICS_PATH"),
                                  path_to_save_plots=env.__getattr__("PLOTS_PATH"),
                                  **collector_config)
    collector.run()


if __name__ == "__main__":
    train_plotter()
