import importlib.metadata as metadata
import importlib.util as pkg

class PackageNotFoundError(Exception):
    """
    Custom exception to handle package not found errors.
    """

    def __init__(self, package_name):
        self.package_name = package_name
        self.message = f"Package '{package_name}' not found."
        self.guide = f'try to install the package using pip eg.pip3 install {package_name}'

    def __str__(self):
        return self.message, self.guide

def is_package_available(pkg_name: str) -> bool:
    """
    Check if a package is available.

    Args:
        pkg_name (str): The name of the package to check.

    Returns:
        bool: True if the package is available, False otherwise.
    """
    return pkg.find_spec(pkg_name) is not None

def get_package_version(pkg_name: str) -> str: # type: ignore
    """
    Get the version of a package.

    Args:
        pkg_name (str): The name of the package to get the version for.

    Returns:
        str: A string indicating whether the package is installed and its version.
    """
    if is_package_available(pkg_name):
        return f'{pkg_name} found [installed] and version is {metadata.version(pkg_name)}'
    
if __name__ == '__main__':
    print(get_package_version('torch'))
    print(is_package_available("requests"))
