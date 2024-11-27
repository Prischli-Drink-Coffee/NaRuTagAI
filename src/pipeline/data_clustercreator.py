from src import path_to_config
from src.script.cluster_collector import ClusterCollector
from src.utils.config_parser import ConfigParser
from src.utils.custom_logging import setup_logging
from env import Env

log = setup_logging()
env = Env()


def data_clustercreator():
    config = ConfigParser.parse(path_to_config())
    collector_config = config.get('ClusterCollector', {})
    collector = ClusterCollector(data_folder=env.__getattr__("DATA_PATH"),
                                 path_to_plots=env.__getattr__("PLOTS_PATH"),
                                 **collector_config)
    collector.run()


if __name__ == "__main__":
    data_clustercreator()
