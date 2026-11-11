"""Compatibility and seed patches for the four official repositories.

Every change below leaves each model, its training loop and its metrics
unchanged. The patches only do two things:

1. Keep 2022-era code running on current numpy (>= 2.0) and pandas (>= 2.0):
   - np.Inf  -> np.inf            (np.Inf was removed in numpy 2.0)
   - df.drop([...], 1)            -> df.drop(columns=[...])
   - s.apply(lambda row: ..., 1)  -> s.apply(lambda row: ...)
   (The last two sit in a branch that only runs when --embed is not 'timeF';
    they are patched anyway so no run can crash on them.)
   - to_offset('t') -> to_offset('min') in utils/timefeatures.py (pandas >= 3.0 removed the
     't' alias for minutes; on older pandas both give the same Minute offset). Only
     Autoformer's ETTm1 configuration passes --freq t.

2. Let LTSF-Linear, Autoformer and iTransformer take a seed from the SEED
   environment variable. Their run scripts hard-code the seed (2021, 2021,
   2023). With SEED unset, the original seed is used, so default behaviour
   is identical to the official code. PatchTST already has --random_seed.

3. Optional switch, OFF by default: if the environment variable P24_NO_TIMESTAMPS=1,
   iTransformer does not receive its calendar tokens (x_mark), exactly as its own code
   already does for the PEMS and Solar datasets. Used only by the 'onzonal_nots' plan
   (RQ2 timestamp check); with the variable unset the official behaviour is unchanged.

Switches added for the final runs (branch final-runs). Each is OFF unless its
environment variable or command-line flag is given; with all of them off, every
computation is the official code plus patches 1-3 above (tested in tests/):

4. P24_ALL_WINDOWS=1 (PatchTST, dual scoring). Only the test pass at the end, in
   exp_main.test(), uses a copy of the official test loader with drop_last=False, and
   prints both the every-window metrics and the metrics of the first floor(n/b)*b windows,
   which are exactly the windows (and predictions) the official loader scores. The
   per-epoch "Test Loss" pass inside train() is NOT touched.
5. P24_EXPORT=1 (PatchTST, LTSF-Linear): also save true.npy next to pred.npy (the official
   line is commented out). iTransformer and Autoformer already save both.
6. --cal_hour / --cal_dow / --cal_holiday (PatchTST+Cal, Objective 5). argparse.SUPPRESS
   defaults, so with no flag the args namespace (and the printed "Args in experiment")
   is unchanged. Adds layers/P24_calendar.py. Embeddings are created after every baseline
   parameter, zero-initialised without drawing random numbers.
7. P24_ITR_CI=1 (iTransformer, Control 2): channel-independent iTransformer.

The script asserts that each expected patch was applied, so a changed
upstream file fails loudly instead of silently.

Usage:  python patch_repos.py <repos_dir>
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "repos")
KIT = Path(__file__).resolve().parent

REPOS = {
    "PatchTST": ROOT / "PatchTST" / "PatchTST_supervised",
    "LTSF-Linear": ROOT / "LTSF-Linear",
    "Autoformer": ROOT / "Autoformer",
    "iTransformer": ROOT / "iTransformer",
}
# (file relative to repo root, original seed line, default seed)
SEED_PATCHES = {
    "LTSF-Linear": ("run_longExp.py", "fix_seed = 2021", 2021),
    "Autoformer": ("run.py", "fix_seed = 2021", 2021),
    "iTransformer": ("run.py", "fix_seed = 2023", 2023),
}

DROP_RE = re.compile(r"\.drop\((\[[^\]]*\]),\s*1\)")
APPLY_RE = re.compile(r"\.apply\((lambda row: row\.[a-z]+(?:\(\))?),\s*1\)")


def patch_compat(repo: Path) -> int:
    n = 0
    for sub in ("data_provider", "utils"):
        for f in (repo / sub).rglob("*.py"):
            s = f.read_text()
            t = s.replace("np.Inf", "np.inf")
            t = DROP_RE.sub(r".drop(columns=\1)", t)
            t = APPLY_RE.sub(r".apply(\1)", t)
            if t != s:
                f.write_text(t)
                n += 1
    return n


FREQ_OLD = "    offset = to_offset(freq_str)\n"
FREQ_NEW = ("    offset = to_offset({'t': 'min', 'T': 'min'}.get(freq_str, freq_str))"
            "  # P24 compat: pandas >= 3 has no 't' alias; same Minute offset\n")


def patch_freq_alias(repo: Path) -> None:
    f = repo / "utils" / "timefeatures.py"
    s = f.read_text()
    if FREQ_NEW in s:
        return  # already patched
    assert s.count(FREQ_OLD) == 1, f"{f}: expected exactly one 'offset = to_offset(freq_str)'"
    f.write_text(s.replace(FREQ_OLD, FREQ_NEW))


def patch_seed(repo: Path, fname: str, old: str, default: int) -> None:
    f = repo / fname
    s = f.read_text()
    new = f"fix_seed = int(os.environ.get('SEED', '{default}'))"
    if new in s:
        return  # already patched
    assert s.count(old) == 1, f"{f}: expected exactly one '{old}'"
    s = s.replace(old, new)
    if not re.search(r"^import os\s*$", s, flags=re.M):
        s = "import os\n" + s
    f.write_text(s)


ITRANS_OLD = "if 'PEMS' in self.args.data or 'Solar' in self.args.data:"
ITRANS_NEW = ("if 'PEMS' in self.args.data or 'Solar' in self.args.data"
              " or os.environ.get('P24_NO_TIMESTAMPS') == '1':")


def patch_itransformer_timestamps(repo: Path) -> None:
    f = repo / "experiments" / "exp_long_term_forecasting.py"
    s = f.read_text()
    if ITRANS_NEW in s:
        return  # already patched
    assert s.count(ITRANS_OLD) == 3, f"{f}: expected 3 timestamp branches"
    assert re.search(r"^import os\s*$", s, flags=re.M), f"{f}: expected 'import os'"
    f.write_text(s.replace(ITRANS_OLD, ITRANS_NEW))


# ------------------------------------------------------------------ final-runs patches
def _sub(f: Path, old: str, new: str, count: int = 1, within: tuple = None) -> None:
    """Replace `old` by `new` in file f, asserting `old` occurs exactly `count` times.

    `within=(start, end)` restricts the edit to the text between the first `start` marker
    and the next `end` marker (used to touch one class of a file only). A patch whose
    `new` text is already present is skipped, so setup can run twice.
    """
    s = f.read_text()
    a, b = 0, len(s)
    if within:
        assert s.count(within[0]) == 1, f"{f}: expected one {within[0]!r}"
        a = s.index(within[0])
        b = s.find(within[1], a + len(within[0]))
        b = len(s) if b < 0 else b
    part = s[a:b]
    if new in part:
        return  # already patched
    n = part.count(old)
    assert n == count, f"{f}: expected {count} occurrence(s) of {old.strip()[:70]!r}, found {n}"
    f.write_text(s[:a] + part.replace(old, new) + s[b:])


ENV = "os.environ.get('{}') == '1'"


def patch_patchtst_dual_scoring(repo: Path) -> None:
    """P24_ALL_WINDOWS=1: score every test window AND report the official-loader subset."""
    f = repo / "exp" / "exp_main.py"
    assert re.search(r"^import os\s*$", f.read_text(), flags=re.M), f"{f}: expected 'import os'"
    # (a) the final test pass gets a copy of the official test loader with drop_last=False.
    #     Same dataset object, batch size, sampler (shuffle=False) and workers; only the last
    #     partial batch is added at the end. train()'s per-epoch pass keeps the official loader.
    _sub(f,
         "    def test(self, setting, test=0):\n"
         "        test_data, test_loader = self._get_data(flag='test')\n",
         "    def test(self, setting, test=0):\n"
         "        test_data, test_loader = self._get_data(flag='test')\n"
         "        if " + ENV.format("P24_ALL_WINDOWS") + ":  # P24 dual scoring: keep the last partial batch\n"
         "            from torch.utils.data import DataLoader\n"
         "            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False,\n"
         "                                     num_workers=self.args.num_workers, drop_last=False)\n")
    # (b) batches now differ in size, so stack by concatenation (same order, same values).
    old_stack = ("        preds = np.array(preds)\n"
                 "        trues = np.array(trues)\n"
                 "        inputx = np.array(inputx)\n"
                 "\n"
                 "        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])\n"
                 "        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])\n"
                 "        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])\n")
    _sub(f, old_stack,
         "        if " + ENV.format("P24_ALL_WINDOWS") + ":  # P24 dual scoring: last batch is shorter\n"
         "            preds = np.concatenate(preds, axis=0)\n"
         "            trues = np.concatenate(trues, axis=0)\n"
         "            inputx = np.concatenate(inputx, axis=0)\n"
         "        else:\n" + "".join("    " + l if l.strip() else l for l in old_stack.splitlines(True)))
    # (c) official-subset metrics: the first floor(n/b)*b windows, printed before the
    #     official every-window line (which p24.py reads as `mse`).
    _sub(f,
         "        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)\n"
         "        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))\n",
         "        if " + ENV.format("P24_ALL_WINDOWS") + ":  # P24 dual scoring: the official loader's windows\n"
         "            k = (len(preds) // self.args.batch_size) * self.args.batch_size\n"
         "            o_mae, o_mse = metric(preds[:k], trues[:k])[:2]\n"
         "            print('P24_OFFICIAL_SUBSET n_scored_official={} n_test={} batch_size={} '\n"
         "                  'mse_official={} mae_official={}'.format(k, len(preds), self.args.batch_size, o_mse, o_mae))\n"
         "        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)\n"
         "        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))\n")


def patch_save_true(repo: Path, rel: str) -> None:
    """P24_EXPORT=1: also save true.npy (the official line is commented out)."""
    f = repo / rel
    assert re.search(r"^import os\s*$", f.read_text(), flags=re.M), f"{f}: expected 'import os'"
    _sub(f,
         "        # np.save(folder_path + 'true.npy', trues)\n",
         "        # np.save(folder_path + 'true.npy', trues)\n"
         "        if " + ENV.format("P24_EXPORT") + ":  # P24 prediction export\n"
         "            np.save(folder_path + 'true.npy', trues)\n")


def patch_patchtst_cal(repo: Path) -> None:
    """PatchTST+Cal: --cal_hour / --cal_dow / --cal_holiday, all off by default."""
    # (a) the new module (calendar embeddings), copied from the kit
    src = (KIT / "p24_calendar.py").read_text()
    dst = repo / "layers" / "P24_calendar.py"
    assert not dst.exists() or dst.read_text() == src, f"{dst} exists and differs from the kit's copy"
    dst.write_text(src)
    # (b) flags; SUPPRESS keeps the namespace unchanged when a flag is not given
    _sub(repo / "run_longExp.py",
         "    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')\n",
         "    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')\n"
         "    # P24 PatchTST+Cal (Objective 5); absent from args unless given\n"
         "    parser.add_argument('--cal_hour', action='store_true', default=argparse.SUPPRESS, help='P24: hour-of-day embedding')\n"
         "    parser.add_argument('--cal_dow', action='store_true', default=argparse.SUPPRESS, help='P24: day-of-week embedding')\n"
         "    parser.add_argument('--cal_holiday', action='store_true', default=argparse.SUPPRESS,\n"
         "                        help='P24: Ontario holiday flag embedding (Dataset_Custom only)')\n")
    # (c) holiday flag column in Dataset_Custom (passed only when --cal_holiday is given)
    dl = repo / "data_provider" / "data_loader.py"
    custom = ("class Dataset_Custom(Dataset):", "\nclass ")
    _sub(dl,
         "                 target='OT', scale=True, timeenc=0, freq='h'):\n",
         "                 target='OT', scale=True, timeenc=0, freq='h', cal_holiday=False):\n"
         "        self.p24_cal_holiday = cal_holiday  # P24 PatchTST+Cal\n", within=custom)
    _sub(dl,
         "        self.data_stamp = data_stamp\n",
         "        if self.p24_cal_holiday:  # P24 PatchTST+Cal: Ontario holiday flag (0/1) as the last column\n"
         "            import holidays\n"
         "            hol = holidays.Canada(subdiv='ON', years=sorted(set(df_stamp.date.dt.year)))\n"
         "            flag = np.array([d.date() in hol for d in df_stamp.date], dtype=float).reshape(-1, 1)\n"
         "            data_stamp = np.concatenate([data_stamp, flag.astype(data_stamp.dtype)], axis=1)\n"
         "        self.data_stamp = data_stamp\n", within=custom)
    _sub(repo / "data_provider" / "data_factory.py",
         "    data_set = Data(\n"
         "        root_path=args.root_path,\n",
         "    data_set = Data(\n"
         "        **({'cal_holiday': True} if getattr(args, 'cal_holiday', False) else {}),  # P24 PatchTST+Cal\n"
         "        root_path=args.root_path,\n")
    # (d) model: build the embeddings last; add them next to W_pos
    m = repo / "models" / "PatchTST.py"
    _sub(m,
         "from layers.PatchTST_layers import series_decomp\n",
         "from layers.PatchTST_layers import series_decomp\n"
         "from layers.P24_calendar import P24CalendarPE, p24_cal_on\n")
    _sub(m,
         "    def forward(self, x):           # x: [Batch, Input length, Channel]\n",
         "        # P24 PatchTST+Cal (off unless a --cal_* flag is given). Built after every baseline\n"
         "        # parameter and zero-initialised without drawing random numbers (layers/P24_calendar.py).\n"
         "        self.p24_cal = None\n"
         "        if p24_cal_on(configs):\n"
         "            assert not self.decomposition, 'P24 PatchTST+Cal supports decomposition=0 only'\n"
         "            self.p24_cal = P24CalendarPE(configs, d_model, self.model.backbone.patch_num, stride, context_window)\n"
         "\n"
         "    def forward(self, x, x_mark=None):           # x: [Batch, Input length, Channel]\n")
    _sub(m,
         "            x = self.model(x)\n",
         "            x = self.model(x) if self.p24_cal is None else self.model(x, self.p24_cal(x_mark))  # P24\n")
    b = repo / "layers" / "PatchTST_backbone.py"
    _sub(b,
         "    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]\n",
         "    def forward(self, z, cal=None):                                                         # z: [bs x nvars x seq_len]\n")
    _sub(b,
         "        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]\n",
         "        z = self.backbone(z) if cal is None else self.backbone(z, cal)                      # P24: cal [bs x patch_num x d_model]\n")
    _sub(b,
         "    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]\n",
         "    def forward(self, x, cal=None) -> Tensor:                                    # x: [bs x nvars x patch_len x patch_num]\n")
    _sub(b,
         "        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]\n",
         "        u = self.dropout(u + self.W_pos) if cal is None else \\\n"
         "            self.dropout(u + self.W_pos + cal.repeat_interleave(n_vars, dim=0))  # P24: same calendar vector for every channel\n")
    # (e) pass batch_x_mark to the model only when a calendar flag is on
    e = repo / "exp" / "exp_main.py"
    _sub(e,
         "        super(Exp_Main, self).__init__(args)\n",
         "        super(Exp_Main, self).__init__(args)\n"
         "        self.p24_cal = any(getattr(args, k, False) for k in ('cal_hour', 'cal_dow', 'cal_holiday'))  # P24\n")
    _sub(e,
         "outputs = self.model(batch_x)\n",
         "outputs = self.model(batch_x) if not self.p24_cal else self.model(batch_x, batch_x_mark)  # P24\n",
         count=8)  # vali x2, train x2, test x2, predict x2


def patch_itransformer_ci(repo: Path) -> None:
    """P24_ITR_CI=1: channel-independent iTransformer (Control 2), calendar tokens kept."""
    f = repo / "model" / "iTransformer.py"
    _sub(f, "import torch\nimport torch.nn as nn\n", "import os\nimport torch\nimport torch.nn as nn\n")
    _sub(f,
         "        self.use_norm = configs.use_norm\n",
         "        self.use_norm = configs.use_norm\n"
         "        self.p24_ci = " + ENV.format("P24_ITR_CI") + "  # P24 Control 2: channel-independent\n")
    _sub(f,
         "        _, _, N = x_enc.shape # B L N\n",
         "        _, _, N = x_enc.shape # B L N\n"
         "        if self.p24_ci:  # P24 Control 2: [B, L, N] -> [B*N, L, 1]; each variate sees only its own calendar tokens\n"
         "            B = x_enc.shape[0]\n"
         "            x_enc = x_enc.permute(0, 2, 1).reshape(B * N, -1, 1)\n"
         "            if x_mark_enc is not None:\n"
         "                x_mark_enc = x_mark_enc.repeat_interleave(N, dim=0)\n")
    _sub(f,
         "        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates\n",
         "        if self.p24_ci:  # P24 Control 2: keep each sequence's variate token, [B*N, S, 1] -> [B, S, N]\n"
         "            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :1].reshape(B, N, -1).permute(0, 2, 1)\n"
         "        else:\n"
         "            dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates\n")


def main() -> None:
    for name, repo in REPOS.items():
        assert repo.is_dir(), f"missing repo {repo}"
        n = patch_compat(repo)
        patch_freq_alias(repo)
        print(f"{name}: compatibility patches in {n} file(s); freq alias 't' -> 'min'")
        if name in SEED_PATCHES:
            patch_seed(repo, *SEED_PATCHES[name])
            print(f"{name}: seed now read from $SEED (default {SEED_PATCHES[name][2]})")
        if name == "iTransformer":
            patch_itransformer_timestamps(repo)
            print("iTransformer: calendar tokens can be switched off with P24_NO_TIMESTAMPS=1 (default: on)")
            patch_itransformer_ci(repo)
            print("iTransformer: channel-independent mode with P24_ITR_CI=1 (default: off)")
        if name == "PatchTST":
            patch_patchtst_dual_scoring(repo)
            print("PatchTST: dual scoring with P24_ALL_WINDOWS=1 (default: off)")
            patch_save_true(repo, "exp/exp_main.py")
            patch_patchtst_cal(repo)
            print("PatchTST: --cal_hour/--cal_dow/--cal_holiday available (default: off)")
        if name == "LTSF-Linear":
            patch_save_true(repo, "exp/exp_main.py")
        if name in ("PatchTST", "LTSF-Linear"):
            print(f"{name}: true.npy saved with P24_EXPORT=1 (default: off)")
    # Check nothing incompatible is left in the code paths we run.
    bad = []
    for name, repo in REPOS.items():
        for sub in ("data_provider", "utils"):
            for f in (repo / sub).rglob("*.py"):
                s = f.read_text()
                if "np.Inf" in s or DROP_RE.search(s) or APPLY_RE.search(s):
                    bad.append(str(f))
    assert not bad, f"unpatched files: {bad}"
    print("All patches applied.")


if __name__ == "__main__":
    main()
