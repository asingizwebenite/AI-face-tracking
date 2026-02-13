from datetime import datetime

class HistoryLogger:
    def __init__(self, identity):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{identity.lower()}_history_{timestamp}.txt"
        self.file = open(filename, "w")
        print(f"[HISTORY FILE CREATED] {filename}")

    def log(self, action, description=""):
        time_now = datetime.now().strftime("%H:%M:%S")
        self.file.write(f"{time_now} | {action} | {description}\n")
        self.file.flush()
