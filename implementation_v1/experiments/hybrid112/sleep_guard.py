"""Scoped logind sleep protection for an isolated long-running diagnostic.

The returned inhibitor descriptor is owned only by the controller, never
inherited by native workers. Exiting or crashing releases it automatically.
No power configuration or GPU driver state is changed.
"""
from contextlib import contextmanager
import os

from jeepney import DBusAddress, MessageType, new_method_call
from jeepney.fds import FileDescriptor
from jeepney.io.blocking import open_dbus_connection


@contextmanager
def prevent_sleep():
    address = DBusAddress('/org/freedesktop/login1', bus_name='org.freedesktop.login1',
                          interface='org.freedesktop.login1.Manager')
    with open_dbus_connection(bus='SYSTEM', enable_fds=True) as connection:
        reply = connection.send_and_get_reply(new_method_call(
            address, 'Inhibit', 'ssss', ('sleep', 'Privacy RAG diagnostic',
            'Fixed-budget private computation and CUDA final comparison', 'block')), timeout=10)
        if (reply.header.message_type != MessageType.method_return or len(reply.body) != 1
                or not isinstance(reply.body[0], FileDescriptor)):
            raise RuntimeError('SLEEP_INHIBITOR_UNAVAILABLE')
        descriptor = reply.body[0]
    with descriptor:
        os.set_inheritable(descriptor.fileno(), False)
        yield
