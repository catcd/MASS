import time


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.job = None

    def start(self, job):
        if job is None:
            return None
        self.start_time = time.time()
        self.job = job
        print("[INFO] {job} started.".format(job=self.job))

    def stop(self):
        if self.job is None:
            return None
        elapsed_time = time.time() - self.start_time
        print("[INFO] {job} finished in {elapsed_time:0.3f} s."
              .format(job=self.job, elapsed_time=elapsed_time))
        self.job = None

class Log:
    verbose = True
    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)