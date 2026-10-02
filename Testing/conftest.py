"""Offline by default, including collection; live embedding checks require opt-in."""
import socket

import pytest


def pytest_addoption(parser):
    parser.addoption('--run-live', action='store_true', help='Allow paid live API tests')


def pytest_configure(config):
    config.addinivalue_line('markers', 'live: requires credentials and paid external APIs')
    if config.getoption('--run-live'):
        return

    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    # Prevent import-time dotenv loading from reading local credentials.
    patch.setattr('dotenv.load_dotenv', lambda *args, **kwargs: False)

    def blocked(*args, **kwargs):
        raise AssertionError('Network access disabled; mock external services in offline tests')

    patch.setattr(socket.socket, 'connect', blocked)
    patch.setattr(socket.socket, 'connect_ex', blocked)
    patch.setattr(socket, 'create_connection', blocked)


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-live'):
        for item in items:
            if item.get_closest_marker('live'):
                item.add_marker(pytest.mark.skip(reason='Live API test; opt in with --run-live'))
