import argparse
import random

PIECES = (
    "⏎ ★ ☆ -7 = yesss sooo naïve in HEY ‘ ’ “ ” w/ https://a.b/c-d ; : + 🎉 a@b.co Lt Col 20-25 $2 Jan. 3.14 "
    "jane.doe@example.com 12:30 1,234 ' % $3.50 ❤ º it.Now ³ } { x² www.x.fr ₂ – — ¥ a.m. p.m. π • 0.05 ( ) "
    "° 12 007 +44 * ± mg/kg Dr. Mr. → million , \" ! ² ¢ the ending St. 5k ½ °C ~ bn x ? 555-123-4567 ≈ α × [ ] "
    "N° € Ĉu 2024 1999-2005 é cat # and/or @ & 2024-05-12 . -42X £ 5 vs. 1. $ … 1st 2nd 3rd 21st $1,000,000 "
    "100% 0.5% 10:45pm 7am -0.25 1/2 3/4 USD km/h e.g. i.e. etc. U.S. NASA A.I. www.example.com/path?q=1 "
    "x@y.org ¿ ¡ ¶ § © ® ™ ‰ µ Ω ∞ √ ≠ ≤ ≥ ÷ 🙂 👍🏽 日本 Привет café résumé coöperate ooooh nooo wheee "
    "9:00 12/25/2024 #1 3x 2.5kg 4K 1080p 0x1F 007Bond ending. word, word; word: word! word? (word) \"word\""
).split()
SEPARATORS = [" ", " ", " ", " ", "", "  ", "\t", " , ", ". ", "! ", "? "]


def strings(n, seed):
    rnd = random.Random(seed)
    out = []
    while len(out) < n:
        k = rnd.randint(1, 14)
        s = ""
        for i in range(k):
            s += rnd.choice(PIECES)
            if i + 1 < k:
                s += rnd.choice(SEPARATORS)
        if rnd.random() < 0.3:
            s += " "
        if rnd.random() < 0.2:
            s = rnd.choice(["", " ", "( ", "“", "-"]) + s
        if s.strip() and s not in out:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("out")
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        for s in strings(a.count, a.seed):
            f.write(s + "\n")


if __name__ == "__main__":
    main()
