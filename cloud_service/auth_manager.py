class AuthManager:
    """
    Manages vehicle authorization based on whitelists and blacklists.
    """

    def __init__(self, whitelist=None, blacklist=None):
        """
        Initializes the AuthManager with optional whitelisted and blacklisted plates.

        :param whitelist: List of whitelisted vehicle plates.
        :param blacklist: List of blacklisted vehicle plates.
        """
        self.white_list = whitelist if whitelist is not None else []
        self.black_list = blacklist if blacklist is not None else []

    def add_blacklisted_plate(self, plate: str):
        """
        Adds a vehicle plate to the blacklist.

        :param plate: The vehicle plate to blacklist.
        """
        if plate not in self.black_list:
            self.black_list.append(plate)

    def add_whitelisted_plate(self, plate: str):
        """
        Adds a vehicle plate to the whitelist.

        :param plate: The vehicle plate to whitelist.
        """
        if plate not in self.white_list:
            self.white_list.append(plate)

    def is_whitelisted(self, plate: str) -> bool:
        """
        Checks if a vehicle plate is in the whitelist.

        :param plate: The vehicle plate to check.
        :return: True if the plate is whitelisted, otherwise False.
        """
        return plate in self.white_list

    def is_blacklisted(self, plate: str) -> bool:
        """
        Checks if a vehicle plate is in the blacklist.

        :param plate: The vehicle plate to check.
        :return: True if the plate is blacklisted, otherwise False.
        """
        return plate in self.black_list

    def get_vehicle_authorization(self, plate: str) -> int:
        """
        Determines the authorization status of a vehicle based on its plate.

        :param plate: The vehicle plate to check.
        :return: 
            -1 if blacklisted (requires special action),
             1 if whitelisted (access granted),
             0 if neither (access denied).
        """
        if self.is_blacklisted(plate):
            return -1
        if self.is_whitelisted(plate):
            return 1
        return 0

