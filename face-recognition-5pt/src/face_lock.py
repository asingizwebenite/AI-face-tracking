class FaceLock:
    def __init__(self, target_name, max_lost=30):
        self.target_name = target_name
        self.locked = False
        self.locked_id = None
        self.lost_frames = 0
        self.max_lost = max_lost
        self.locked_bbox = None

    def try_lock(self, identity, similarity, threshold, bbox):
        if not self.locked:
            if identity == self.target_name and similarity > threshold:
                self.locked = True
                self.locked_id = identity
                self.locked_bbox = bbox
                print(f"[LOCKED] {identity}")
                return True
        return False

    def update_tracking(self, identity, bbox):
        if not self.locked:
            return False

        if identity == self.locked_id:
            self.locked_bbox = bbox
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        if self.lost_frames > self.max_lost:
            print("[UNLOCKED] Face lost")
            self.locked = False
            self.locked_id = None
            return False

        return True
