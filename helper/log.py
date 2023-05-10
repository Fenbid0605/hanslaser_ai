import logging

log_format = '[%(asctime)s] %(levelname)s %(message)s'
datetime_format = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=datetime_format)

log = logging.getLogger(__name__)
