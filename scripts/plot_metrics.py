
import matplotlib.pyplot as plt, json, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()
    data = json.load(open(args.metrics))
    plt.plot(data["reward"], label="Reward")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.legend(); plt.tight_layout(); plt.savefig("training_curve.png")
if __name__ == "__main__": main()
