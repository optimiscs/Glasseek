import ipaddress
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation_backend'))
from tools import url_safety

def addr(ip):
    family = socket.AF_INET6 if ':' in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, 0, '', (ip, 0))

class ExternalFetchDnsSafetyTest(unittest.TestCase):
    def check(self, local, verified, url='https://example.test/a'):
        with patch.object(url_safety.socket, 'getaddrinfo', return_value=[addr(ip) for ip in local]), \
             patch.object(url_safety, '_doh_public_addresses', return_value=[ipaddress.ip_address(ip) for ip in verified]):
            return url_safety.is_safe_external_fetch_url(url)

    def test_poisoned_ipv6_rechecked_against_public_dns(self):
        self.assertTrue(self.check(['8.8.8.8', '2001::1'], ['208.80.154.224', '2001:df2:e500:ed1a::1']))

    def test_uses_one_consistent_local_dns_snapshot(self):
        with patch.object(url_safety.socket, 'getaddrinfo', return_value=[addr('8.8.8.8'), addr('2001::1')]) as lookup, \
             patch.object(url_safety, '_doh_public_addresses', return_value=[ipaddress.ip_address('208.80.154.224')]):
            self.assertTrue(url_safety.is_safe_external_fetch_url('https://example.test/'))
            self.assertEqual(lookup.call_count, 1)

    def test_private_local_address_still_blocked(self):
        self.assertFalse(self.check(['127.0.0.1', '2001::1'], ['8.8.8.8']))

    def test_private_verified_answer_still_blocked(self):
        self.assertFalse(self.check(['8.8.8.8', '2001::1'], ['8.8.8.8', '10.0.0.1']))

    def test_literal_and_metadata_blocked(self):
        self.assertFalse(self.check(['2001::1'], ['8.8.8.8'], 'http://127.0.0.1/'))
        self.assertFalse(self.check(['2001::1'], ['8.8.8.8'], 'http://metadata.google.internal/'))

    def test_dns_failure_fails_closed(self):
        with patch.object(url_safety.socket, 'getaddrinfo', return_value=[addr('8.8.8.8'), addr('2001::1')]), \
             patch.object(url_safety, '_doh_public_addresses', side_effect=TimeoutError):
            self.assertFalse(url_safety.is_safe_external_fetch_url('https://example.test/'))

if __name__ == '__main__': unittest.main()
