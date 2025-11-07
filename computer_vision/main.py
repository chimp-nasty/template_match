from manager.client import Client
from utils.helpers import resolve_logic_class
from utils.hwnd_utils import hwnds_for
import config as cfg


# Interim Startup sequence - this is setup for some kind of multiboxing
if __name__ == "__main__":
    LogicClass = resolve_logic_class(cfg.LOGIC_CLASS)
    hwnds = hwnds_for(cfg.PROCESS)
    hwnd = hwnds[0] if hwnds else None

    clients = []
    
    if len(hwnds) > 1:
        for hwnd in hwnds:
            client = Client(target_fps=cfg.FPS, hwnd=hwnd, logic=LogicClass(need_rgb=True, need_hsv=False), debug=cfg.DEBUG, display=False)
            clients.append(client)
    
    else:
        client = Client(target_fps=cfg.FPS, hwnd=hwnd, logic=LogicClass(need_rgb=True, need_hsv=False), debug=cfg.DEBUG, display=True)
        clients.append(client)

    for i, client in enumerate(clients, start=1):
        print(f"Starting Client {i}: hwnd={client.hwnd}")
        client.start()