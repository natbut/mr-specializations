import socket

with socket.socket() as s:
    s.connect(('localhost', 9999))
    s.sendall( b'planning_trigger')