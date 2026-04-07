from line_profiler import LineProfiler

"""
To use the line profiler, put the @profile decorator above the function you want to profile
"""

profiler = LineProfiler()


def profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)

    return inner


def print_stats():
    profiler.print_stats()


def dump_stats(file_name):
    profiler.dump_stats(f"profile/{file_name}.lprof")
