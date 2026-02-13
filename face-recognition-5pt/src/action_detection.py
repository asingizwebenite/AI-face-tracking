
prev_nose_x = None
blink_counter = 0
# laugh_counter = 0 
def detect_head_movement(nose_x, threshold):
    global prev_nose_x
    action = None

    if prev_nose_x is not None:
        diff = nose_x - prev_nose_x
        if diff > threshold:
            action = "Moved Right"
        elif diff < -threshold:
            action = "Moved Left"

    prev_nose_x = nose_x
    return action


def detect_blink(eye_top, eye_bottom, threshold):
    global blink_counter
    ear = abs(eye_top[1] - eye_bottom[1])

    if ear < threshold:
        blink_counter += 1
    else:
        if blink_counter > 2:
            blink_counter = 0
            return "Blink"
        blink_counter = 0
    return None


def detect_smile(mouth_left, mouth_right, threshold):
    width = abs(mouth_right[0] - mouth_left[0])
    if width > threshold:
        return "Smile"
    return None


def detect_laugh(
    nose,
    mouth_left,
    mouth_right,
    smile_threshold,
    drop_threshold,
    frames_required=5
):
    """
    Laugh = very wide smile + mouth drops below nose for several frames
    Uses ONLY 5 points.
    """
    global laugh_counter

    mouth_width = abs(mouth_right[0] - mouth_left[0])

    mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2

    mouth_drop = mouth_mid_y - nose[1]

    if mouth_width > smile_threshold and mouth_drop > drop_threshold:
        laugh_counter += 1
        if laugh_counter >= frames_required:
            laugh_counter = 0
            return "Laugh"
    else:
        laugh_counter = 0

    return None