"""Small runtime compatibility shims for Hydra entrypoints."""
import argparse
import sys


def patch_argparse_help_for_hydra():
    """Allow Hydra's lazy help object on Python versions with stricter argparse.

    Python 3.14 validates `Action.help` with string containment. Hydra 1.x uses
    a local lazy help object for shell completion, which is accepted by older
    argparse versions but raises before the script reaches the Hydra main. This
    restores the older behavior only for non-string help objects.
    """
    if sys.version_info < (3, 14):
        return
    if getattr(argparse.ArgumentParser, "_hydra_lazy_help_compat", False):
        return

    original_check_help = argparse.ArgumentParser._check_help

    def _check_help(self, action):
        help_value = getattr(action, "help", None)
        if help_value is not None and not isinstance(help_value, str):
            return
        return original_check_help(self, action)

    argparse.ArgumentParser._check_help = _check_help
    argparse.ArgumentParser._hydra_lazy_help_compat = True
