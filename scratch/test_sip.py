import socket
import time

def test_sip(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(3.0)
    
    msg = (
        "OPTIONS sip:{}@{} SIP/2.0\r\n"
        "Via: SIP/2.0/UDP 127.0.0.1:5060;branch=z9hG4bK-test\r\n"
        "Max-Forwards: 70\r\n"
        "To: <sip:{}@{}>\r\n"
        "From: <sip:test@127.0.0.1>;tag=12345\r\n"
        "Call-ID: test-12345\r\n"
        "CSeq: 1 OPTIONS\r\n"
        "Contact: <sip:test@127.0.0.1:5060>\r\n"
        "Accept: application/sdp\r\n"
        "Content-Length: 0\r\n\r\n"
    ).format(ip, ip, ip, ip)
    
    try:
        sock.sendto(msg.encode('utf-8'), (ip, port))
        print(f"Sent OPTIONS to {ip}:{port}")
        data, addr = sock.recvfrom(2048)
        print(f"Received from {addr}:")
        print(data.decode('utf-8'))
    except socket.timeout:
        print("Timeout: No response from SIP server")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()

test_sip("yumi.pstn.twilio.com", 5060)
