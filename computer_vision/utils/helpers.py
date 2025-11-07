import importlib


def resolve_logic_class(dotted: str):
    mod_path, cls_name = dotted.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)