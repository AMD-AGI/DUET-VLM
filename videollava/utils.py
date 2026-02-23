import datetime
import logging
import logging.handlers
import os
import sys
from torch import nn
import numpy as np
import requests
from typing import List, Optional, Dict, Any, Sequence

from videollava.constants import LOGDIR


server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

def order_pick_k(lst, k):
    if len(lst) <= k:
        return lst
    rng = np.random.random(len(lst))
    index = np.argsort(rng)[:k]
    index_sort = sorted(index)
    new_lst = [lst[i] for i in index_sort]
    print(
        f"WARNING: total file: {len(lst)}, random pick: {k}."
        f" (ignored)"
    )
    return new_lst


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


# ============== Longest common contiguous subsequence (for PDrop) ==================
def to_list(x):
    """Convert torch tensor / numpy array / list-like to Python list of ints."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def longest_common_subarray(qqs: Sequence[int],
                            prompt: Sequence[int]
                            ) -> Optional[Dict[str, Any]]:
    """
    Full-DP (O(n*m) time & space) longest contiguous common subarray.
    If multiple matches have the same max length, return the one with the
    earliest start index in `prompt` (the second argument).

    Args:
      qqs: sequence of token ids (list/tuple/torch.Tensor/numpy.ndarray)
      prompt: sequence of token ids (list/tuple/torch.Tensor/numpy.ndarray)

    Returns:
      dict with keys:
        - length
        - start_in_qqs_0based, end_in_qqs_0based
        - start_in_prompt_0based, end_in_prompt_0based
        - start_in_qqs_1based, end_in_qqs_1based
        - start_in_prompt_1based, end_in_prompt_1based
        - match (list of token ids)
      or None if no common contiguous subarray.
    """
    if not isinstance(qqs, list):
        qqs = to_list(qqs)
    if not isinstance(prompt, list):
        prompt = to_list(prompt)

    if not qqs or not prompt:
        return None

    n, m = len(qqs), len(prompt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    max_len = 0
    end_a = -1
    end_b = -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if qqs[i - 1] == prompt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                cur_len = dp[i][j]
                cur_start_b = j - cur_len
                if cur_len > max_len:
                    max_len = cur_len
                    end_a = i - 1
                    end_b = j - 1
                elif cur_len == max_len and cur_len > 0:
                    prev_start_b = end_b - max_len + 1
                    if cur_start_b < prev_start_b:
                        end_a = i - 1
                        end_b = j - 1
            else:
                dp[i][j] = 0

    if max_len == 0:
        return None

    start_a = end_a - max_len + 1
    start_b = end_b - max_len + 1
    match = qqs[start_a:end_a + 1]

    return {
        "length": max_len,
        "start_in_qqs_0based": start_a,
        "end_in_qqs_0based": end_a,
        "start_in_prompt_0based": start_b,
        "end_in_prompt_0based": end_b,
        "start_in_qqs_1based": start_a + 1,
        "end_in_qqs_1based": end_a + 1,
        "start_in_prompt_1based": start_b + 1,
        "end_in_prompt_1based": end_b + 1,
        "match": match
    }
# ===================================================================================
