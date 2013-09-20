"""File to interface with Top-level interfaces to XFlow"""

# Versions:
#  2013-08-18 @jdahm   : First version

# ------- Modules required -------
# Used for parsing input from files
import re


class xf_JobFile(dict):
    """The file used to initialize XFlow"""

    @classmethod
    def read(cls, fname):
        """
        Reads a jobfile from disk.

        >>> J = xf_JobFile.read("../examples/poisson.job")
        """
        data = {}
        with open(fname, "r") as f:
            # Loop over lines in the file
            for l in f:
                # Try to match a key-val pair
                m = re.match(r"\s*(?P<key>[^\s=]+)\s*=\s*(?P<val>[^\s\n]+)", l)
                # Add if available
                if m is not None: data[m.group("key")] = m.group("val")
        return cls(data)

    def write(self, fname):
        """
        Writes a jobfile to disk.
        """
        with open(fname, "w") as f:
            for (key, val) in self.data.items():
                f.write("{} = {}\n".format(key, val));


if __name__ == "__main__":
    import doctest
    doctest.testmod()
