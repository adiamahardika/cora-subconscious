def register_events():
    from . import socketio

    @socketio.on('connect')
    def handle_connect():
        print("Client connected")

    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")

    @socketio.on('inference')
    def handle_inference(data):
        print(data)
        socketio.emit('inference', data, broadcast=True)

    @socketio.on('inference_response')
    def handle_inference_response(data):
        print(data)
        socketio.emit('inference_response', data, broadcast=True)