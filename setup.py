import setuptools


setuptools.setup(
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*'],
    },
)

