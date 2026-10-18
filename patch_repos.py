"""Compatibility and seed patches for the four official repositories.

Every change below leaves each model, its training loop and its metrics
unchanged. The patches only do two things:

1. Keep 2022-era code running on current numpy (>= 2.0) and pandas (>= 2.0):
   - np.Inf  -> np.inf            (np.Inf was removed in numpy 2.0)
   - df.drop([...], 1)            -> df.drop(columns=[...])
   - s.apply(lambda row: ..., 1)  -> s.apply(lambda row: ...)
   (The last two sit in a branch that only runs when --embed is not 'timeF';
    they are patched anyway so no run can crash on them.)

2. Let LTSF-Linear, Autoformer and iTransformer take a seed from the SEED
   environment variable. Their run scripts hard-code the seed (2021, 2021,
   2023). With SEED unset, the original seed is used, so default behaviour
   is identical to the official code. PatchTST already has --random_seed.

The script asserts that each expected patch was applied, so a changed
upstream file fails loudly instead of silently.

Usage:  python patch_repos.py <repos_dir>
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "repos")

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


def main() -> None:
    for name, repo in REPOS.items():
        assert repo.is_dir(), f"missing repo {repo}"
        n = patch_compat(repo)
        print(f"{name}: compatibility patches in {n} file(s)")
        if name in SEED_PATCHES:
            patch_seed(repo, *SEED_PATCHES[name])
            print(f"{name}: seed now read from $SEED (default {SEED_PATCHES[name][2]})")
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
