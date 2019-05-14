try:
    from .utils import ddict
    from .datasets import load_dataset
    from .mnist_multitask import load_multitask_dataset

except ImportError as e:
    import sys
    print('''Could not import submodules (exact error was: %s).''' % e, file=sys.stderr)


__all__ = [
    'ddict', 'load_dataset', 'load_multitask_dataset', 'Flatten'
]
