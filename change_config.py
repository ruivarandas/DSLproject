import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-beat")
    args = parser.parse_args()

    with open("config.json", "r") as f:
        data = json.load(f)

    data["heartbeat"] = str(args.beat)

    with open("config.json", "w") as f:
        json.dump(data, f)
