import sys
import os

if len(sys.argv) < 2:
    print("Usage: python main.py [train|eval|api]")
    exit()

cmd = sys.argv[1]

if cmd == "train":
    os.system("python -m training.train")

elif cmd == "eval":
    os.system("python -m training.evaluate")

elif cmd == "api":
    os.system("uvicorn api.app:app --reload")

else:
    print("Invalid command")
