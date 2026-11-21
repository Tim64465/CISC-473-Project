"""P24 PatchTST+Cal: calendar-aware positional encoding (Objective 5).

patch_repos.py copies this file into the official PatchTST repo as
PatchTST_supervised/layers/P24_calendar.py. It is used only when one of the flags
--cal_hour, --cal_dow, --cal_holiday is given to run_longExp.py.

Learned hour-of-day (24 x d), day-of-week (7 x d) and holiday-flag (2 x d) embeddings are
added to every patch token next to PatchTST's own position embedding W_pos
(layers/PatchTST_backbone.py, TSTiEncoder.forward). They are shared across channels and
indexed by the timestamp of each patch's first time step.

Randomness: nn.Embedding(n, d) would call reset_parameters() -> normal_(), which draws
from the global torch RNG and would shift every later random number (dropout masks,
shuffling). Passing _weight=torch.zeros(n, d) skips reset_parameters(), so building this
module draws no random number, and its output is exactly 0 until training updates it.
"""
import os

import torch
from torch import nn

from utils.timefeatures import time_features_from_frequency_str

FLAGS = ("cal_hour", "cal_dow", "cal_holiday")


def p24_cal_on(args) -> bool:
    return any(getattr(args, k, False) for k in FLAGS)


def _zero_embedding(n: int, d: int) -> nn.Embedding:
    return nn.Embedding(n, d, _weight=torch.zeros(n, d))  # no reset_parameters(): no RNG draw


class P24CalendarPE(nn.Module):
    def __init__(self, configs, d_model: int, patch_num: int, stride: int, context_window: int):
        super().__init__()
        self.use_hour = bool(getattr(configs, "cal_hour", False))
        self.use_dow = bool(getattr(configs, "cal_dow", False))
        self.use_holiday = bool(getattr(configs, "cal_holiday", False))
        assert configs.embed == "timeF", "P24 PatchTST+Cal reads the timeF calendar features"
        # Column of each feature in x_mark, from the official feature list for this --freq
        # (utils/timefeatures.py:85-108; e.g. freq 'h' -> HourOfDay, DayOfWeek, DayOfMonth, DayOfYear).
        names = [type(f).__name__ for f in time_features_from_frequency_str(configs.freq)]
        self.n_time_features = len(names)
        self.hour_col = names.index("HourOfDay") if self.use_hour else None
        self.dow_col = names.index("DayOfWeek") if self.use_dow else None
        self.patch_num, self.stride = patch_num, stride
        # First step of patch j is j*stride (z.unfold(..., step=stride) in PatchTST_backbone.forward);
        # the 'end' padding only appends copies after the last step, so every first step is real.
        assert (patch_num - 1) * stride < context_window
        if self.use_hour:
            self.hour = _zero_embedding(24, d_model)
        if self.use_dow:
            self.dow = _zero_embedding(7, d_model)
        if self.use_holiday:
            self.holiday = _zero_embedding(2, d_model)
        if os.environ.get("P24_CAL_FREEZE") == "1":  # validation only (plan V3): embeddings stay exactly 0
            for p in self.parameters():
                p.requires_grad_(False)

    @staticmethod
    def hour_index(x):
        # HourOfDay = index.hour / 23.0 - 0.5   (utils/timefeatures.py:38)  =>  hour = round((x + 0.5) * 23)
        return torch.round((x + 0.5) * 23).long()

    @staticmethod
    def dow_index(x):
        # DayOfWeek = index.dayofweek / 6.0 - 0.5   (utils/timefeatures.py:45)  =>  dow = round((x + 0.5) * 6), Monday = 0
        return torch.round((x + 0.5) * 6).long()

    def forward(self, x_mark):  # x_mark: [bs x seq_len x n_mark] -> [bs x patch_num x d_model]
        assert x_mark.shape[-1] == self.n_time_features + int(self.use_holiday), x_mark.shape
        first = torch.arange(self.patch_num, device=x_mark.device) * self.stride
        m = x_mark[:, first, :]  # calendar features at the first step of each patch
        out = 0
        if self.use_hour:
            out = out + self.hour(self.hour_index(m[..., self.hour_col]))
        if self.use_dow:
            out = out + self.dow(self.dow_index(m[..., self.dow_col]))
        if self.use_holiday:  # 0/1 flag appended as the last column by Dataset_Custom (P24 patch)
            out = out + self.holiday(torch.round(m[..., -1]).long())
        return out
