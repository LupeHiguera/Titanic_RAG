"""Verify default test isolation without contacting any external endpoint."""
import socket

import dotenv
import pytest


@pytest.fixture(autouse=True)
def offline_only(request):
    if request.config.getoption('--run-live'):
        pytest.skip('Offline guard is deliberately disabled in live mode')


def test_dotenv_does_not_read_local_credentials(tmp_path, monkeypatch):
    name = 'TITANIC_OFFLINE_GUARD_SENTINEL'
    monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / 'synthetic.env'
    env_file.write_text(f'{name}=synthetic-value\n')
    assert dotenv.load_dotenv(env_file) is False
    import os

    assert name not in os.environ


@pytest.mark.parametrize('method', ['connect', 'connect_ex', 'create_connection'])
def test_outbound_connection_methods_are_blocked(method):
    # A loopback address with an invalid service port avoids external traffic
    # even if the guard regresses; an OS connection error must not pass this test.
    address = ('127.0.0.1', 0)
    with pytest.raises(AssertionError, match='Network access disabled'):
        if method == 'create_connection':
            socket.create_connection(address, timeout=0.1)
        else:
            with socket.socket() as sock:
                getattr(sock, method)(address)
