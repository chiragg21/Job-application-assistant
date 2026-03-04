import os
import re
import configparser
from dotenv import load_dotenv

def _cast_value(value: str):
    """Cast a string value to float, int, list, or leave as str."""
    # List: comma-separated values
    if ',' in value:
        return [_cast_value(v.strip()) for v in value.split(',') if v.strip()]

    # Float (must check before int to catch "-2.5", "0.1")
    try:
        f = float(value)
        return int(f) if f == int(f) and '.' not in value else f
    except ValueError:
        pass

    return value


def get_config_dict():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path, override=True)

    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    config.read(config_path)

    res = {}
    for section in config.sections():
        res[section] = {}
        for key, value in config.items(section):
            expanded = os.path.expandvars(value).strip()
            res[section][key] = _cast_value(expanded)

    return res