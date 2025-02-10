import logging

log_format = "%(asctime)s | %(levelname)-5s %(funcName)-30s %(message)s"
log_level = logging.DEBUG
#log_level = logging.NOTSET

'''
logging.basicConfig(
level=log_level,
#filename=f"logs/log.log",
filename=None,
format=log_format,
filemode="w",
)

LOGGER = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
console_handler.setLevel(log_level)
LOGGER.addHandler(console_handler)
'''

logging.basicConfig(format=log_format, level=log_level)
LOGGER = logging.getLogger(__name__)