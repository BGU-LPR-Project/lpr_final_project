class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float = 1.0):
        """
        Represents a rectangular bounding box.

        Args:
            x (int): Top-left x-coordinate.
            y (int): Top-left y-coordinate.
            width (int): Width of the box.
            height (int): Height of the box.
            confidence (float): Confidence score (default: 1.0).
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def intersects_with(self, other: 'BoundingBox') -> bool:
        """
        Checks if this box intersects with another box.

        Args:
            other (BoundingBox): The other box to compare.

        Returns:
            bool: True if boxes overlap.
        """
        return not (
            self.x + self.width < other.x or
            self.x > other.x + other.width or
            self.y + self.height < other.y or
            self.y > other.y + other.height
        )

    def merge_with(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        Merges this box with another, creating a new bounding box that 
        tightly contains both.

        Args:
            other (BoundingBox): The other box to merge with.

        Returns:
            BoundingBox: A new bounding box that encompasses both.
        """
        new_x = min(self.x, other.x)
        new_y = min(self.y, other.y)
        new_w = max(self.x + self.width, other.x + other.width) - new_x
        new_h = max(self.y + self.height, other.y + other.height) - new_y
        return BoundingBox(new_x, new_y, new_w, new_h, max(self.confidence, other.confidence))

    def is_inside(self, other: 'BoundingBox') -> bool:
        """
        Checks if this box is completely inside another box.

        Args:
            other (BoundingBox): The containing box.

        Returns:
            bool: True if this box is inside the other.
        """
        return (
            self.x >= other.x and
            self.y >= other.y and
            self.x + self.width <= other.x + other.width and
            self.y + self.height <= other.y + other.height
        )
