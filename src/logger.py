# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import logging


def print_args(function):
    def _print_args(*args, **kw):
        logging.info(function.__name__, args, kw)
        return function(*args, **kw)
    return print_args