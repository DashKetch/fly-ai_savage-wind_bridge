"""Pong against a real fruit-fly brain (the fly.ai connectome).

    python main.py check      does the court reach the fly's neurons? how fast does the brain step?
    python main.py train      let the fly practice against a ball machine and learn from it
    python main.py play       you vs the fly (it keeps learning from every point)
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "train", "play"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                       help="where to run the connectome (default: GPU if there is one)")
        p.add_argument("--fresh", action="store_true", help="ignore what the fly learned before")
    sub.choices["train"].add_argument("--rallies", type=int, default=60, help="practice balls (default 60)")
    sub.choices["play"].add_argument("--no-learn", action="store_true", help="freeze the fly while you play")
    args = ap.parse_args()

    from fly_player import FlyPlayer            # imported late so --help works without flybrain installed

    fly = FlyPlayer(device=args.device)
    if args.cmd != "check" and not args.fresh:
        fly.load()

    if args.cmd == "check":
        from training import check
        check(fly)
    elif args.cmd == "train":
        from training import practice
        practice(fly, rallies=args.rallies)
    else:
        if not fly.readout.useful:
            print("the fly has not learned anything yet: its paddle stays idle until it has seen a few balls.\n"
                  "(python main.py train  gives it a head start)")
        from ui import play
        play(fly, learn=not args.no_learn)


if __name__ == "__main__":
    main()
