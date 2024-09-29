from src import path_to_config
from src.utils.config_parser import ConfigParser
from src.script.data_collector import RutubeVideoCollector
from src.utils.custom_logging import setup_logging
from env import Env

log = setup_logging()


def data_downloader():
    env = Env()
    config = ConfigParser.parse(path_to_config())

    collector_config = config.get('RutubeVideoCollector', {})
    collector = RutubeVideoCollector(data_folder=env.__getattr__("DATA_PATH"),
                                     **collector_config)
    collector.run()


if __name__ == "__main__":
    data_downloader()
