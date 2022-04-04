import datetime
import enum
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import testpath

PROXYSCRIPT = shutil.which("qstat_proxy.sh")

EXAMPLE_QSTAT_CONTENT = """
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
15399.s034-lcam   DROGON-1         combert                     0 H hb120
15400.s034-lcam   DROGON-2         barbert                     0 R hb120
15402.s034-lcam   DROGON-3         foobert                     0 E hb120
""".strip()

PROXYFILE_FOR_TESTS = "proxyfile"

MOCKED_QSTAT_BACKEND = (
    # NB: This mock does not support the job id as an argument.
    'import time; time.sleep(0.5); print("""'
    + EXAMPLE_QSTAT_CONTENT
    + '""")'
)
MOCKED_QSTAT_BACKEND_FAILS = "import sys; sys.exit(1)"
MOCKED_QSTAT_BACKEND_LOGGING = (
    "import uuid; open('log/' + str(uuid.uuid4()), 'w').write('.'); "
    + MOCKED_QSTAT_BACKEND
)


@pytest.mark.parametrize("jobid", [15399, 15400, 15402])
def test_recent_proxyfile_exists(tmpdir, jobid):
    os.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND):
        result = subprocess.run(
            [PROXYSCRIPT, str(jobid), PROXYFILE_FOR_TESTS],
            check=True,
            capture_output=True,
        )
    assert str(jobid) in str(result.stdout)
    if sys.platform.startswith("darwin"):
        # On Darwin, the proxy script falls back to the mocked backend which is
        # not feature complete for this test:
        assert len(result.stdout.splitlines()) == 5
    else:
        assert len(result.stdout.splitlines()) == 3


def test_proxyfile_not_exists(tmpdir):
    """If there is no proxy file, the backend should be called"""
    os.chdir(tmpdir)
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND):
        result = subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
            capture_output=True,
        )
    if not sys.platform.startswith("darwin"):
        assert Path(PROXYFILE_FOR_TESTS).exists()
    assert "15399" in str(result.stdout)
    if sys.platform.startswith("darwin"):
        # (the mocked backend is not feature complete)
        assert len(result.stdout.splitlines()) == 5
    else:
        assert len(result.stdout.splitlines()) == 3


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_missing_backend_script(tmpdir):
    """If a cache file is there, we will use it, but if not, and there is no
    backend, we fail"""
    os.chdir(tmpdir)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
        )
    # Try again with cached file present
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    subprocess.run(
        [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
        check=True,
    )


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_recent_proxyfile_locked(tmpdir):
    """If the proxyfile is locked in the OS, and it is not too old, we should use it.

    NB: This test relies on the proxy script utilizing 'flock', which is possibly
    an implementation detail."""
    os.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    with open(PROXYFILE_FOR_TESTS, encoding="utf-8") as proxy_fd:
        fcntl.flock(proxy_fd, fcntl.LOCK_EX)
        # Ensure that if we fall back to the backend, we fail the test:
        with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND_FAILS):
            subprocess.run(
                [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                check=True,
                capture_output=False,
            )
        fcntl.flock(proxy_fd, fcntl.LOCK_UN)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_old_proxyfile_exists(tmpdir):
    """If the proxyfile is there, but old, acquire the lock, fix the cache file,
    and return the correct result."""
    os.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(
        EXAMPLE_QSTAT_CONTENT.replace("15399", "25399"),
        encoding="utf-8"
        # (if this proxyfile is used, it will fail the test)
    )
    # Manipulate mtime of the file so the script thinks it is old:
    eleven_seconds_ago = datetime.datetime.now() - datetime.timedelta(seconds=11)
    os.utime(
        PROXYFILE_FOR_TESTS,
        (eleven_seconds_ago.timestamp(), eleven_seconds_ago.timestamp()),
    )
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND):
        result = subprocess.run(
            [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
            check=True,
            capture_output=True,
        )
        print(result)
        assert Path(PROXYFILE_FOR_TESTS).exists()
        assert "15399" in str(result.stdout)
        assert len(result.stdout.splitlines()) == 3


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_old_proxyfile_locked(tmpdir):
    """If the proxyfile is locked in the OS, and it is too old to use, we fail hard

    NB: This rest relies on the proxy script utilizing 'flock', which is possibly
    an implementation detail."""
    os.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    # Manipulate mtime of the file so the script thinks it is old:
    eleven_seconds_ago = datetime.datetime.now() - datetime.timedelta(seconds=11)
    os.utime(
        PROXYFILE_FOR_TESTS,
        (eleven_seconds_ago.timestamp(), eleven_seconds_ago.timestamp()),
    )

    with open(PROXYFILE_FOR_TESTS, encoding="utf-8") as proxy_fd:
        fcntl.flock(proxy_fd, fcntl.LOCK_EX)
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(
                [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                check=True,
            )
        fcntl.flock(proxy_fd, fcntl.LOCK_UN)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_proxyfile_not_existing_but_locked(tmpdir):
    """This is when another process has locked the output file for writing, but
    it has not finished yet (and the file is thus empty). The proxy should fail in
    this situation."""
    os.chdir(tmpdir)
    with open(PROXYFILE_FOR_TESTS, "w", encoding="utf-8") as proxy_fd:
        fcntl.flock(proxy_fd, fcntl.LOCK_EX)
        assert os.stat(PROXYFILE_FOR_TESTS).st_size == 0
        with pytest.raises(subprocess.CalledProcessError):
            result = subprocess.run(
                [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                check=True,
                capture_output=True,
            )
            print(str(result.stdout))
            print(str(result.stderr))
        fcntl.flock(proxy_fd, fcntl.LOCK_UN)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_many_concurrent_qstat_invocations(tmpdir):
    """Run many qstat invocations simultaneously, with a mocked backend qstat
    script that logs how many times it is invoked, then we assert that it has
    not been called often.

    If this test has enough invocations (or the hardware is slow enough), this test
    will also pass the timeout in the script yielding a rerun of the backend. In that
    scenario there is a risk for a race condition where the cache file is blank when
    another process is reading from it. This test also assert that failures are only
    allowed to happen in a sequence from the second invocation (the first process
    succeeds because it calls the backend, the second one fails because there is no
    cache file and it is locked.)

    This test will dump something like
    01111111111000000000000000000000000000000000000 to stdout
    where each digit is the return code from the proxy script. Only one sequence of 1's
    is allowed after the start, later on there should be no failures (that would be
    errors from race conditions.)
    """
    starttime = time.time()
    invocations = 400
    sleeptime = 0.01  # seconds. Lower number increase probability of race conditions.
    # (the mocked qstat backend sleeps for 0.5 seconds to facilitate races)
    cache_timeout = 2  # This is CACHE_TIMEOUT in the shell script
    assert invocations * sleeptime > cache_timeout  # Ensure race conditions can happen

    os.chdir(tmpdir)
    Path("log").mkdir()  # The mocked backend writes to this directory
    subprocesses = []
    with testpath.MockCommand("qstat", python=MOCKED_QSTAT_BACKEND_LOGGING):
        for _ in range(invocations):
            subprocesses.append(
                subprocess.Popen(
                    [PROXYSCRIPT, "15399", PROXYFILE_FOR_TESTS],
                    stdout=subprocess.DEVNULL,
                )
            )
            time.sleep(sleeptime)

        class CacheState(enum.Enum):
            # Only consecutive transitions are allowed.
            FIRST_INVOCATION = 0
            FIRST_HOLDS_FLOCK = 1
            CACHE_EXISTS = 2

        state = None
        for _, process in enumerate(subprocesses):
            process.wait()
            if state is None:
                if process.returncode == 0:
                    state = CacheState.FIRST_INVOCATION
                assert state is not None, "First invocation should not fail"

            elif state == CacheState.FIRST_INVOCATION:
                assert process.returncode == 1
                # The proxy should fail in this scenario, and ERTs queue
                # manager must retry later.
                state = CacheState.FIRST_HOLDS_FLOCK

            elif state == CacheState.FIRST_HOLDS_FLOCK:
                if process.returncode == 1:
                    # Continue waiting until the cache is ready
                    pass
                if process.returncode == 0:
                    state = CacheState.CACHE_EXISTS

            else:
                assert state == CacheState.CACHE_EXISTS
                assert (
                    process.returncode == 0
                ), "Check for race condition if AssertionError"

            print(process.returncode, end="")
        print("\n")

    # Allow a limited set of backend runs. We get more backend runs the
    # slower the iron.
    time_taken = time.time() - starttime
    backend_runs = len(list(Path("log").iterdir()))
    print(
        f"We got {backend_runs} backend runs from "
        f"{invocations} invocations in {time_taken:.2f} seconds."
    )

    # We require more than one backend run because there is race condition we need
    # to test for. Number of backend runs should then be relative to the time taken
    # to run the test (plus 3 for slack)
    assert 1 < backend_runs < int(time_taken / cache_timeout) + 3


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_no_such_job_id(tmpdir):
    """Ensure we replicate qstat's error behaviour, yielding error
    if a job id does not exist."""

    os.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [PROXYSCRIPT, "10001", PROXYFILE_FOR_TESTS],
            check=True,
        )


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No flock on MacOS")
def test_no_argument(tmpdir):
    """qstat with no arguments lists all jobs. So should the proxy."""
    os.chdir(tmpdir)
    Path(PROXYFILE_FOR_TESTS).write_text(EXAMPLE_QSTAT_CONTENT, encoding="utf-8")
    result = subprocess.run(
        [PROXYSCRIPT, "", PROXYFILE_FOR_TESTS],
        check=True,
        capture_output=True,
    )
    assert len(result.stdout.splitlines()) == 5