import logging
import logging.config as log_config

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "simple": {
            "format": '"%(asctime)s"\t"%(pathname)s"\t"%(module)s"\t"%(funcName)s"\t"%(lineno)d"\t"[%(levelname)s]"\t-\t"%(message)s"'
        }
    },
    "handlers": {
        "file": {
            "level": "DEBUG",
            "formatter": "simple",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "logs/classification.log",
            "when": "midnight",
            "encoding": "utf-8",
            "backupCount": 7,
        },
        "stdout": {
            "level": "DEBUG",
            "formatter": "simple",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["file", "stdout"],
        "level": "DEBUG",
        "propagate": True,
    },
    "loggers": {
        "text-classification-toolkit": {
            "handlers": ["file", "stdout"],
            "level": "DEBUG",
            "propagate": False,
        },
        "PaddleNLP": {
            "handlers": ["file", "stdout"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

log_config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("text-classification-toolkit")
