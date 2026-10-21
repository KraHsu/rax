"""Stdio MCP server the agent CLI starts: forwards to the session's robot service socket.

The launcher starts the robot service (services.controller.frontend --socket) once per
launch; it owns the episodes and simulators and outlives any one agent process, so the
CLI's own tool calls and the agent's Python scripts share one serialized server.
"""
import os
import socket
import sys
import threading


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ['RAX_MCP_SOCKET']
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(path)

        def upload():
            try:
                while block := os.read(0, 65536):
                    connection.sendall(block)
            finally:
                try:
                    connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
        threading.Thread(target=upload, daemon=True).start()
        while block := connection.recv(65536):
            sys.stdout.buffer.write(block)
            sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()
