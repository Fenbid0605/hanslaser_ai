import asyncio
import json

from helper.log import log


class Cursor:
    status = True
    conn = None
    x: 0
    y: 0

    def __init__(self, rpc):
        self.conn = rpc

    async def get_position(self, websocket):
        self.status = True
        while self.status:
            await asyncio.sleep(0.3)
            self.x, self.y = self.conn.root.get_cursor()
            log.info('The cursor position of the IPC is: (%s,%s)' % (self.x, self.y))
            await websocket.send(json.dumps({'type': 'GetCursorPosition', 'x': self.x, 'y': self.y}))

    def stop(self):
        self.status = False
