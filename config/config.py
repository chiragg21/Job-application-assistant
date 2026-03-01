import os
import configparser
from dotenv import load_dotenv

def get_config_dict():
    # Load .env explicitly
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path, override=True)

    # Disable interpolation (IMPORTANT FIX)
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str  # preserve case

    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    config.read(config_path)

    res = {}
    for section in config.sections():
        res[section] = {}
        for key, value in config.items(section):
            # Expand ${VAR} using environment variables
            res[section][key] = os.path.expandvars(value)

    return res


# if __name__ == "__main__":
#     c = get_config_dict()
#     print(c)