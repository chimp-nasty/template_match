# dummy game logic
class DummyLogic:
    def __init__(self, need_rgb, need_hsv):
        self.data = None
        self.need_rgb = need_rgb
        self.need_hsv = need_hsv

    def start(self):
        if not self.data:
            return { "status": False, "message": "Calibration Failed" }
        else:
            return { "status": True, "message": "Calibration Success" }
    
    # game specific img processing
    def update_data(self, data):
        self.data = data

    def execute(self):
        rgb = self.data.get("rgb")
        if rgb is not None and len(rgb) > 0:
            i = rgb[0]