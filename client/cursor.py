import asyncio

from helper.log import log


class Cursor:
    status = True
    conn = None

    def __init__(self, rpc):
        self.conn = rpc

    async def get_position(self):
        self.status = True
        while self.status:
            await asyncio.sleep(0.3)
            x, y = self.conn.root.get_cursor()
            log.info('The cursor position of the IPC is: (%s,%s)' % (x, y))

    def stop(self):
        self.status = False
