import signal


class TimeoutException(Exception):  # Custom exception class
    pass


def break_after(*, seconds: int = 0, minutes: int = 0, hours: int = 0):
    seconds += minutes * 60 + hours * 3600

    # timeout decorator
    def timeout_handler(signum, frame):  # Custom signal handler
        raise TimeoutException

    def function(function):
        def wrapper(*args, **kwargs):
            nonlocal seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                res = function(*args, **kwargs)
                signal.alarm(0)  # Clear alarm
                return res
            except TimeoutException:
                hours, remainder = divmod(seconds, 3600)  # noqa823
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"Timeout after {hours}h:{minutes}m:{seconds}s",
                    function.__name__,
                )
            return

        return wrapper

    return function
