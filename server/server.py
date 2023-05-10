import pyautogui
import rpyc
from rpyc.utils.server import ThreadedServer  # or ForkingServer
from helper.log import log


class Service(rpyc.Service):
    def __init__(self):
        self._conn = None

    def on_connect(self, conn):
        self._conn = conn

    def exposed_ping(self):
        log.info('ping')
        return 'pong'

    def exposed_get_cursor(self):
        return pyautogui.position()


if __name__ == '__main__':
    import socket

    hostname = socket.gethostname()
    log.info(hostname)
    server = ThreadedServer(Service, port=18888)
    server.start()
