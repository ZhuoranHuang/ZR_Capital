"""YAML config loader.

Reads `settings.yaml` and `strategies/*.yaml` from this directory and returns
plain dicts. No env-var overlay, no caching — keep it boring until we have
a real reason to grow.
"""
import os

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_yaml(path):
    if not os.path.isabs(path):
        path = os.path.join(CONFIG_DIR, path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_settings():
    return load_yaml('settings.yaml')


def load_strategy(name):
    return load_yaml(os.path.join('strategies', f'{name}.yaml'))
