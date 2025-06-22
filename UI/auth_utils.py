import json
from pathlib import Path
from werkzeug.security import check_password_hash

# Define the correct path to the user data file
USER_DATA_PATH = Path(__file__).parent / "data" / "user.dat"

def load_user(filepath=USER_DATA_PATH):
    """
    Load user credentials from a JSON file.

    :param filepath: Path to the user data file
    :return: Tuple of (username, password_hash)
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data["username"], data["password_hash"]

def verify_credentials(input_username, input_password, filepath=USER_DATA_PATH):
    """
    Verifies username and password against stored hash.

    :param input_username: Username entered by user
    :param input_password: Password entered by user
    :param filepath: Path to user data file
    :return: True if credentials match, else False
    """
    try:
        username, password_hash = load_user(filepath)
        if input_username == username and check_password_hash(password_hash, input_password):
            return True
    except Exception as e:
        print(f"[ERROR] Failed to verify credentials: {e}")
    return False