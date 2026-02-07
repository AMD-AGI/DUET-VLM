import datetime
import logging
import logging.handlers
import os
import sys

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

# ============== Longest common contiguous subsequence ==================
from typing import List, Optional, Dict, Any, Sequence

def to_list(x):
    """Convert torch tensor / numpy array / list-like to Python list of ints."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.tolist()
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(x, np.ndarray):
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
    # dp[i][j] length of longest common suffix ending at a[i-1], b[j-1]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    max_len = 0
    end_a = -1
    end_b = -1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if qqs[i - 1] == prompt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                cur_len = dp[i][j]
                cur_start_b = j - cur_len  # 0-based start in prompt
                if cur_len > max_len:
                    max_len = cur_len
                    end_a = i - 1
                    end_b = j - 1
                elif cur_len == max_len and cur_len > 0:
                    # tie-break: prefer smallest start index in prompt (earliest in prompt)
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

def longest_common_subarray_DEPRECATED(a: List[int],
                            b: List[int],
                            tie_break: str = "earliest_prompt"
                           ) -> Optional[Dict[str, Any]]:
    """
    Find the longest contiguous subarray common to lists `a` and `b`.

    Args:
      a, b: input lists (e.g. token id lists).
      tie_break: one of {"earliest_prompt", "earliest_qqs"} used when
                 there are multiple matches of the same (max) length.
                 - "earliest_prompt": prefer the match with smallest start index in `b`.
                 - "earliest_qqs"  : prefer the match with smallest start index in `a`.

    Returns:
      dict with keys:
        - length: int (number of tokens in match)
        - start_in_qqs_0based, end_in_qqs_0based
        - start_in_prompt_0based, end_in_prompt_0based
        - start_in_qqs_1based, end_in_qqs_1based
        - start_in_prompt_1based, end_in_prompt_1based
        - match: List[int] (the matching subarray)
      or None if length == 0 (no contiguous match).
    Complexity: O(len(a) * len(b)) time, O(len(b)) extra memory.
    """
    if not a or not b:
        return None

    n, m = len(a), len(b)
    dp = [0] * (m + 1)  # dp[j] length of suffix match ending at a[i-1], b[j-1]
    max_len = 0
    end_a = -1
    end_b = -1

    for i in range(1, n + 1):
        # iterate b backwards so dp[j-1] is previous iteration's value
        for j in range(m, 0, -1):
            if a[i - 1] == b[j - 1]:
                dp[j] = dp[j - 1] + 1
                cur_len = dp[j]
                cur_end_a = i - 1
                cur_end_b = j - 1
                if cur_len > max_len:
                    max_len = cur_len
                    end_a, end_b = cur_end_a, cur_end_b
                elif cur_len == max_len and cur_len > 0:
                    # tie-break: check earliest start according to preference
                    cur_start_a = cur_end_a - cur_len + 1
                    cur_start_b = cur_end_b - cur_len + 1
                    prev_start_a = end_a - max_len + 1
                    prev_start_b = end_b - max_len + 1
                    if tie_break == "earliest_prompt":
                        if cur_start_b < prev_start_b:
                            end_a, end_b = cur_end_a, cur_end_b
                    elif tie_break == "earliest_qqs":
                        if cur_start_a < prev_start_a:
                            end_a, end_b = cur_end_a, cur_end_b
                    else:
                        raise ValueError("tie_break must be 'earliest_prompt' or 'earliest_qqs'")
            else:
                dp[j] = 0

    if max_len == 0:
        return None

    start_a = end_a - max_len + 1
    start_b = end_b - max_len + 1
    match = a[start_a:end_a + 1]
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

# ============================================

# ============= TODO: HACKS ==================
def extract_between_sequences(input_ids, start_seq, end_seq):
    """
    Extract token IDs between two marker sequences in input_ids.
    """
    def find_subsequence(seq, sub):
        """Return the index of the first occurrence of sub in seq, or -1."""
        for i in range(len(seq) - len(sub) + 1):
            if all(a==b for a, b in zip(seq[i:i+len(sub)], sub)):
                return i
        return -1

    # Find start and end indices
    start_idx = find_subsequence(input_ids, start_seq)
    end_idx = find_subsequence(input_ids, end_seq)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Start or end sequence not found")

    # Extract everything between
    return input_ids[start_idx + len(start_seq): end_idx], start_idx + len(start_seq), end_idx
# ============================================


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

# -------------------------
# Example usage (with your sample lists):
# -------------------------
if __name__ == "__main__":
    qqs = [1, 1724, 338, 278, 1024, 310, 278, 784, 2592, 4318, 29973, 13,
           29909, 29889, 22559, 13, 29933, 29889, 1570, 7904, 28401, 13,
           29907, 29889, 7861, 356, 7935, 13, 29928, 29889, 17589, 609]

    prompt = [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082,
              20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892,
              322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155,
              29889, 3148, 1001, 29901, 529, 3027, 29958, 13, 5618, 338,
              278, 1024, 310, 278, 784, 2592, 4318, 29973, 13, 29909, 29889,
              22559, 13, 29933, 29889, 1570, 7904, 28401, 13, 29907, 29889,
              7861, 356, 7935, 13, 29928, 29889, 17589, 609, 13, 22550,
              411, 278, 2984, 29915, 29879, 5497, 515, 278, 2183, 19995,
              4153, 29889, 319, 1799, 9047, 13566, 29901]

    res = longest_common_subarray(qqs, prompt, tie_break="earliest_prompt")
    if res:
        print("Longest contiguous match found:")
        print(f" length = {res['length']}")
        print(f" match  = {res['match']}")
        print(f" qqs  (0-based): {res['start_in_qqs_0based']} .. {res['end_in_qqs_0based']}")
        print(f" prm  (0-based): {res['start_in_prompt_0based']} .. {res['end_in_prompt_0based']}")
        print(f" qqs  (1-based): {res['start_in_qqs_1based']} .. {res['end_in_qqs_1based']}")
        print(f" prm  (1-based): {res['start_in_prompt_1based']} .. {res['end_in_prompt_1based']}")
    else:
        print("No contiguous match found.")
