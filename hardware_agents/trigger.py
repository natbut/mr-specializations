import argparse
import socket
import struct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--type", type=str, required=True, help="update, plan, or coordinate")
    parser.add_argument("--content", nargs='+', type=float, default=None, help="relevant message info")
    parser.add_argument("--robot_id", type=int, required=True, help="ID of the robot to send the message to")

    args = parser.parse_args()

    # Base port for each trigger type
    base_ports = {
        "plan": 10000,
        "update": 11000,
        "coordinate": 12000
    }

    if args.type not in base_ports:
        raise ValueError("Invalid trigger_type. Must be one of: plan, update, coordinate.")

    # Each robot gets a unique port by adding robot_id to the base port
    port = base_ports[args.type] + args.robot_id

    with socket.socket() as s:
        s.connect(('localhost', port))
        if args.type == "plan":
            s.sendall(b'planning_trigger')
        elif args.type == "update":
            content_byt = struct.pack(f'<{2}f', *args.content)
            s.sendall(content_byt)
        elif args.type == "coordinate":
            s.sendall(b'coordinate_trigger')