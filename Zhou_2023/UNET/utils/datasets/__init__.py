from .cocostuff import CocoStuff10k, CocoStuff164k,surf


def get_dataset(name):
    return {"cocostuff10k": CocoStuff10k, "cocostuff164k": CocoStuff164k,"surf":surf}[name]
