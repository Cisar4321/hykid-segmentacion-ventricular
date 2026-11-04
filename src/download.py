import os
import sys

import synapseclient
import synapseutils

from .paths import RAW

DATA = "syn68545325"
FILES = ["syn68545862", "syn68545861", "syn68545852"]


def main():
    token = os.environ.get("SYNAPSE_AUTH_TOKEN") or input("Synapse token: ").strip()
    syn = synapseclient.Synapse()
    syn.login(authToken=token)
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        syn.get(f, downloadLocation=str(RAW))
    if "--meta" not in sys.argv:
        synapseutils.syncFromSynapse(syn, DATA, path=str(RAW / "data"))
    print("Listo:", RAW)


if __name__ == "__main__":
    main()
