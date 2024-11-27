from src import path_to_config
from src.script.graph_collector import GraphNodeCollector
from src.utils.config_parser import ConfigParser
from src.utils.custom_logging import setup_logging
from env import Env

log = setup_logging()
env = Env()


def data_graphcreator():
    config = ConfigParser.parse(path_to_config())
    collector_config = config.get('GraphNodeCollector', {})
    collector = GraphNodeCollector(data_folder=env.__getattr__("DATA_PATH"),
                                   **collector_config)
    collector.run()


if __name__ == "__main__":
    data_graphcreator()
