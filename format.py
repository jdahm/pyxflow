import sys
import os
import shutil
import glob
import subprocess as sp


def format_C(fname):
    # Astyle should already copy the file to .orig
    shutil.copyfile(fname, fname + ".orig")

    sp.call(["astyle", "--mode=c", "--style=linux",
             "--indent=spaces=4", "--break-blocks",
             "--pad-oper", "--remove-comment-prefix", fname])


def format_python(fname, agressive=True):
    # autopep8 does not back up the file by default
    shutil.copyfile(fname, fname + ".orig")

    sp.call(["autopep8", "--aggressive", "--in-place", fname])


def format_pyxflow():
    C_files = [
        "pyxflow/px_All.c", "pyxflow/px_All.h",
        "pyxflow/px_DataSet.c", "pyxflow/px_DataSet.h",
        "pyxflow/px_Geom.c", "pyxflow/px_Geom.h",
        "pyxflow/px_Mesh.c", "pyxflow/px_Mesh.h",
        "pyxflow/px_Plot.c", "pyxflow/px_Plot.h"]

    Py_files = [
        "format.py",
        "pyxflow/__init__.py",
        "pyxflow/All.py",
        "pyxflow/DataSet.py",
        "pyxflow/Geom.py",
        "pyxflow/Mesh.py",
        "pyxflow/build.py",
        "pyxflow/setup.py"]

    for f in C_files:
        print "Formatting {}".format(f)
        format_C(f)
    for f in Py_files:
        print "Formatting {}".format(f)
        format_python(f)


def clean_pyxflow():
    files = glob.glob("*.orig") + glob.glob("*~") + \
        glob.glob("pyxflow/*.orig") + glob.glob("pyxflow/*~") + \
        glob.glob("pyxflow/lib/*~")
    for f in files:
        print "Removing {}".format(f)
        os.remove(f)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] in ["--clean", "-clean", "clean"]:
            clean_pyxflow()
    else:
        format_pyxflow()
