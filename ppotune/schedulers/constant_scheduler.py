from ppotune.schedulers.scheduler import Scheduler

def constant_scheduler(value: float) -> Scheduler:
    constant_scheduler_fn = lambda _: value

    return Scheduler(constant_scheduler_fn)
