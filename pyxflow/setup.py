from distutils.core import setup, Extension

xflow_home = '/home/dalle/usr/local/xflow'

module1 = Extension('px_Mesh',
    include_dirs = [xflow_home + '/include'],
    libraries = ['xfSerial','CompressibleNS'],
    library_dirs = [xflow_home + '/lib'],
    sources = ['Meshmodule.c'])

setup(
    name = 'px_Mesh',
    version = '1.0',
    description = 'This package is an interface for xf_Mesh.c',
    ext_modules = [module1])
