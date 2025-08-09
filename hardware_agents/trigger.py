import argparse
import socket
import struct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--trigger_type", type=str, required=True, help="update, plan, or coordinate")
    parser.add_argument("--content", nargs='+', type=float, default=None, help="relevant message info")

    args = parser.parse_args()

    with socket.socket() as s:
        
        if args.trigger_type == "plan":
            s.connect(('localhost', 9999))
            s.sendall( b'planning_trigger')
        elif args.trigger_type == "update":
            s.connect(('localhost', 9998))
            # s.sendall( b'update_trigger')
            content_byt = struct.pack(f'<{2}f', *args.content)
            s.sendall(content_byt)
        elif args.trigger_type == "coordinate":
            s.connect(('localhost', 9997))
            s.sendall( b'coordinate_trigger')