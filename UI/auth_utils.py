import json
from pathlib import Path
from werkzeug.security import check_password_hash

USER_DATA_PATH = Path(__file__).parent / "data" / "user.dat"

def load_users(filepath=USER_DATA_PATH):
    """
    Loads all users and their credentials from the JSON file.
    :return: List of user dicts
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("users", [])

def verify_credentials(input_username, input_password, filepath=USER_DATA_PATH):
    """
    Verifies credentials and returns role if correct.
    :return: User role if valid, otherwise None
    """
    try:
        users = load_users(filepath)
        for user in users:
            if user["username"] == input_username and check_password_hash(user["password_hash"], input_password):
                return user["role"]
    except Exception as e:
        print(f"[ERROR] Failed to verify credentials: {e}")
    return None
