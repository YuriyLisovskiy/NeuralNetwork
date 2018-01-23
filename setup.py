import os
import sys
from distutils.sysconfig import get_python_lib
from setuptools import setup

CURRENT_PYTHON_VERSION = sys.version_info[:2]
REQUIRED_PYTHON_VERSION = (3, 5)

NAME = 'neural_network'

if CURRENT_PYTHON_VERSION < REQUIRED_PYTHON_VERSION:
    sys.stderr.write("""
Unsupported Python version!
This version of '{}' requires Python {}.{}, but you're trying to
install it on Python {}.{}.""".format(NAME, *REQUIRED_PYTHON_VERSION, *CURRENT_PYTHON_VERSION))
    sys.exit(1)

existing_path = None

overlay_warning = False
if "install" in sys.argv:
    lib_paths = [get_python_lib()]
    if lib_paths[0].startswith("/usr/lib/"):
        lib_paths.append(get_python_lib(prefix="/usr/local"))
    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "neural_network"))
        if os.path.exists(existing_path):
            overlay_warning = True
            break

DESCRIPTION = 'Feed forward neural network with back propagation learning algorithm'
LONG_DESCRIPTION = 'README.md'
URL = 'https://github.com/YuriyLisovskiy/NeuralNetwork'
EMAIL = 'yuralisovskiy98@gmail.com'
AUTHOR = 'Yuriy Lisovskiy'
VERSION = '0.1.0'
LICENCE = 'BSD'
REQUIRED = ['numpy']
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'License :: OSI Approved :: {} License'.format(LICENCE),
    'Operating System :: Microsoft :: Windows :: Windows 7',
    'Operating System :: Microsoft :: Windows :: Windows 8',
    'Operating System :: Microsoft :: Windows :: Windows 8.1',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Communications :: Email',
    'Natural Language :: English',
    'Natural Language :: Ukrainian',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

setup(
    name=NAME,
    version=VERSION,
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON_VERSION),
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    ong_description=LONG_DESCRIPTION,
    license=LICENCE,
    py_modules=[NAME],
    include_package_data=True,
    install_requires=REQUIRED,
    classifiers=CLASSIFIERS
)

if overlay_warning:
    sys.stderr.write("""
WARNING!
You have just installed '{}' over top of an existing
installation, without removing it first. Because of this,
your install may now include extraneous files from a
previous version that have since been removed from
'{}'. This is known to cause a variety of problems. You
should manually remove the %(existing_path)s
directory and re-install '{}'.""".format(NAME, NAME, NAME) % {"existing_path": existing_path})
