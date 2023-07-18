import rpyc
import asyncio
import websockets

from cursor import Cursor
from helper.log import log


class ClientService(rpyc.Service):
    def __init__(self):
        self._conn = None

    def on_connect(self, conn):
        self._conn = conn
        # asyncio.run(self.ping())

    async def ping(self):
        while True:
            await asyncio.sleep(5)
            log.info(self._conn.root.ping())

    def exposed_get_cursor_location(self):
        return


conn = rpyc.connect('192.168.31.186', 18888, service=ClientService)


async def echo(websocket):
    gcp = Cursor(conn)
    async for message in websocket:
        if message == 'GetCursorPosition':
            _ = asyncio.create_task(gcp.get_position(websocket))
        elif message == 'StopGetCursorPosition':
            gcp.stop()


async def ws():
    async with websockets.serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever


asyncio.run(ws())
