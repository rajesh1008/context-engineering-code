import os, subprocess, sys, pathlib
here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))

# The 34 fixture PDFs and truth.json are not stored in the repository; they are
# rebuilt deterministically (reportlab invariant mode) by make_corpus.py.
if not (here / "corpus" / "truth.json").exists() or not (here / "corpus" / "HO-3_2022-01.pdf").exists():
    subprocess.run([sys.executable, str(here / "make_corpus.py")], cwd=str(here), check=True,
                   stdout=subprocess.DEVNULL)
