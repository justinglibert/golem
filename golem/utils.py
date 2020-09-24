
"""
Attributes:
    default_logger: The default global logger.

"""
import colorlog
from logging import INFO
import time


class FakeLogger(object):
    def setLevel(self, level):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def exception(self, msg, *args, exc_info=True, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass

    def log(self, level, msg, *args, **kwargs):
        pass


_default_handler = colorlog.StreamHandler()
_default_handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(process)s:[%(asctime)s] <%(levelname)s>:%(name)s:%(message)s"))

default_logger = colorlog.getLogger("default_logger")
default_logger.addHandler(_default_handler)
default_logger.setLevel(INFO)
default_logger.propagate = False
fake_logger = FakeLogger()


class SimpleProfiler(object):
    def __init__(self):
        self.start = time.time()
        self.category_start = None
        self.current_category = None
        self.table = {}

    def __call__(self, category):
        if category not in self.table:
            self.table[category] = 0
        self.current_category = category
        self.category_start = time.time()
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_time = time.time() - self.category_start
        self.table[self.current_category] += total_time

    def __repr__(self):
        total_time = time.time() - self.start
        lines = "\n"
        for k in self.table.keys():
            lines += "{}: {}s and {} percent of total time\n".format(
                k, self.table[k] / 1000, self.table[k] / total_time * 100)
        return lines
