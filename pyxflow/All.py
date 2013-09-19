import re

class xf_JobFile(dict):
    """The file used to initialize xflow."""

    @classmethod
    def read(cls, filename):
        """Reads a jobfile from disk.

        J = xf_JobFile.read("../examples/poisson.job")
        """
        data = {}
        with open(filename, "r") as f:
            for l in f:
                m = re.match(r"\s*(?P<key>[^\s=]+)\s*=\s*(?P<val>[^\s\n]+)", l)
                if m is not None: data[m.group("key")] = m.group("val")
        return cls(data)

    def write(self, filename):
        """Writes a jobfile to disk."""
        with open(filename, "w") as f:
            for (key, val) in self.data.items():
                f.write("{} = {}\n".format(key, val));


if __name__ == "__main__":
    import doctest
    doctest.testmod()
