import logging


class EdusetLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        formatter = logging.Formatter(fmt="%(asctime)s\tfile: %(name)s\t[%(levelname)s]: %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.addHandler(console_handler)
        self.setLevel(logging.DEBUG)
