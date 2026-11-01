"""Listing 3.9: Equation 3.2, KV-cache bytes per token, for an illustrative architecture."""
from common import tee

LAYERS, KV_HEADS, D_HEAD, BYTES = 32, 8, 128, 2


def main():
    per = 2 * LAYERS * KV_HEADS * D_HEAD * BYTES
    print(f"bytes per token: 2 x {LAYERS} x {KV_HEADS} x {D_HEAD} x {BYTES} = {per:,}")
    for n in (1_046, 14_463, 60_000, 200_000):
        print(f"{n:>10,} tokens -> {n * per / 2**30:6.2f} GiB per sequence")


if __name__ == "__main__":
    tee("3.9_kv_memory", main)
