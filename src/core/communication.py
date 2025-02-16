import socket

class RoboCommunication:
    def __init__(self, ip="192.168.0.10", port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, port)

    def send_message(self, msg):
        self.sock.sendto(msg.encode('utf-8'), self.addr)

    def receive_message(self):
        data, addr = self.sock.recvfrom(1024)
        return data.decode('utf-8')
