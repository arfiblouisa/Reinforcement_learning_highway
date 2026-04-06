import json
import logging.config
from pathlib import Path
import gymnasium as gym

from rl_agents.configuration import Configurable

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
        "detailed": {"format": "[%(name)s:%(levelname)s] %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "standard", "level": "INFO"}
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}

def configure(config=None, gym_level=logging.INFO):
    """
    Configure logging for rl-agents with gymnasium
    """
    if config:
        if isinstance(config, str):
            with Path(config).open() as f:
                config = json.load(f)
        Configurable.rec_update(logging_config, config)
    logging.config.dictConfig(logging_config)
    # Gymnasium ne possède plus gym.logger
    logging.getLogger("gymnasium").setLevel(gym_level)


def add_file_handler(file_path):
    """
    Add a file handler to the root logger.
    """
    # Crée le handler de fichier avec le formatter 'detailed'
    handler_name = f"file_{file_path.name}"
    file_handler = {
        handler_name: {
            "class": "logging.FileHandler",
            "filename": str(file_path),
            "level": "DEBUG",
            "formatter": "detailed",
            "mode": 'w'
        }
    }

    # Met à jour logging_config
    logging_config["handlers"].update(file_handler)
    logging_config["root"]["handlers"].append(handler_name)

    logging.config.dictConfig(logging_config)
