class BoundingBox:
    def __init__(self, x: int, y: int, width: int, height: int, confidence: float = 1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence

    def intersects_with(self, other: 'BoundingBox') -> bool:
        return not (
            self.x + self.width < other.x or
            self.x > other.x + other.width or
            self.y + self.height < other.y or
            self.y > other.y + other.height
        )

    def merge_with(self, other: 'BoundingBox') -> 'BoundingBox':
        new_x = min(self.x, other.x)
        new_y = min(self.y, other.y)
        new_w = max(self.x + self.width, other.x + other.width) - new_x
        new_h = max(self.y + self.height, other.y + other.height) - new_y
        return BoundingBox(new_x, new_y, new_w, new_h, max(self.confidence, other.confidence))