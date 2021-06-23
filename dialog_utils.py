import regex as re
import numpy as np
import random
import logging


def safe_clean_text(text):
    """ This is a safe text cleaning procedure for all datasets
    """
    # weird words
    text = text.replace("¡ª", "")

    # handle \\\'t \\\'ve
    text = text.replace("\\", "")
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # fix puncutations
    text = text.replace(" ?", "?")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")

    return text


class DialogFragmentSampler:
    def __init__(self, max_tokens=1024, max_turns=20):
        """Sample dialog fragments from a dialog
        """
        self.max_num_tokens = max_tokens - 1
        self.max_num_turns = max_turns

    def __call__(self, dialog):
        """dialog is a dict which has key "token_ids"
        """
        dialog_fragment = {}

        lengths = np.array([len(item) for item in dialog['token_ids']])

        # if the entire dialog is smaller than the max len
        if lengths.sum() <= self.max_num_tokens:
            return dialog

        cumsum_len = lengths.cumsum()
        reverse_cumsum_len = cumsum_len[::-1]

        # based on the reverse cumsum, we can have a range to select from
        start_turns = np.arange(len(reverse_cumsum_len)
                               )[reverse_cumsum_len > self.max_num_tokens]
        # remove odd numbers
        start_turns = [idx for idx in start_turns if idx % 2 == 0]
        # randomly choose one
        try:
            random_start_turn = random.choice(start_turns)
        except:
            breakpoint()
        cumsum_len = np.concatenate([[0], cumsum_len], axis=0)
        new_cumsum_len = cumsum_len - cumsum_len[random_start_turn]

        # find the maximum end turn (only odd turn)
        for i in reversed(range(len(new_cumsum_len))):
            if i % 2 == 1 and new_cumsum_len[i] < self.max_num_tokens:
                random_end_turn = i
                break

        random_end_turn = min(
            random_end_turn, random_start_turn + self.max_num_turns - 1
        )

        dialog_fragment["token_ids"] = dialog['token_ids'][random_start_turn:
                                                           random_end_turn]

        assert sum(
            [len(item) for item in dialog_fragment["token_ids"]]
        ) < self.max_num_tokens

        return dialog_fragment