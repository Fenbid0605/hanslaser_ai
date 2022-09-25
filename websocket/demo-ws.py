from threading import Lock

from flask import Flask
from flask_socketio import SocketIO

async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins=['http://localhost:3000'])
thread = None
thread_lock = Lock()


@socketio.on('connect', namespace='/status')
def status():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(target=background_thread)


def background_thread():
    while True:
        socketio.sleep(5)
        socketio.emit('status_change', {'data': 'Step0'}, namespace='/status')
        socketio.sleep(10)
        socketio.emit('status_change', {'data': 'Step1'}, namespace='/status')
        socketio.sleep(10)
        socketio.emit('status_change', {'data': 'Step2'}, namespace='/status')
        socketio.sleep(20)
        socketio.emit('status_change', {'data': 'Step3'}, namespace='/status')
        socketio.sleep(20)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='localhost', port=3001)
