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

def get_package_version(pkg_name: str) -> str:
    """
    Get the version of a package.

    Args:
        pkg_name (str): The name of the package to get the version for.

    Returns:
        str: A string indicating whether the package is installed and its version.
    """
    if is_package_available(pkg_name):
        return f'{pkg_name} found [installed] and version is {metadata.version(pkg_name)}'

def is_torch_available() -> bool:
    """
    Check if PyTorch is available.

    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    return is_package_available("torch")

def torch_version() -> str:
    """
    Get the version of PyTorch.

    Returns:
        str: A string indicating the version of PyTorch.
    """
    if is_torch_available():
        return f'torch version {get_package_version("torch")}'
    else:
        raise PackageNotFoundError('torch')

def is_tensorflow_available() -> bool:
    """
    check if tensorflow is available

    returns:
    bool: True if tensorflow is available, False otherwise
    """
    return is_package_available("tensorflow")

def tensorflow_version() -> str:
    '''
    get the version of tensorflow
    
    returns:
    str: A string indicating the version of tensorflow
    '''
    if is_tensorflow_available():
        return get_package_version("tensorflow")
    else:
        raise PackageNotFoundError('tensorflow')

def is_tiktoken_available() -> bool:
    """
    Check if tiktoken is available.

    Returns:
        bool: True if tiktoken is available, False otherwise.
    """
    return is_package_available("tiktoken")

def tiktoken_version() -> str:
    """
    Get the version of tiktoken.

    Returns:
        str: A string indicating the version of tiktoken.
    """
    if is_tiktoken_available():
        return get_package_version('tiktoken')
    else:
        raise PackageNotFoundError('tiktoken')

def is_torch_cuda_available() -> bool:
    """
    Check if CUDA for PyTorch is available.

    Returns:
        bool: True if CUDA for PyTorch is available, False otherwise.
    """
    if is_torch_available():
        import torch
        return torch.cuda.is_available()
    else:
        return False

def list_cuda_dev() -> list:
    if is_torch_cuda_available():
        from torch import cuda
        return cuda.device_count
def is_numpy_available() -> bool:
    '''
    Check if numpy is installed or not

    returns:
       bool: True if numpy is installed, False otherwise
    '''
    return is_package_available('numpy')

def numpy_version() -> str:
    '''
    if numpy is installed then return version of numpy installed
    else PackageNotFoundError is raised
    '''
    if is_numpy_available():
        return get_package_version('numpy')
    else:
        raise PackageNotFoundError('numpy')

def is_pandas_available() -> bool:
    '''
    Check if pandas is installed or not

    returns:
    bool: True if pandas is installed, False otherwise
    '''
    return is_package_available('pandas')

def pandas_version() -> str:
    '''
    if pandas is installed then return version of pandas installed
    else PackageNotFounderror is raised
    '''
    if is_pandas_available():
        return get_package_version('pandas')
    else:
        raise PackageNotFounderror('pandas')

def is_safetensors_available() -> bool:
    """
    Check if safetensors are available.

    Returns:
        bool: True if safetensors are available, False otherwise.
    """
    return is_package_available("safetensors")

def safetensors_version() -> str:
    """
    Get the version of safetensors.

    Returns:
        str: A string indicating the version of safetensors.
    """
    if is_safetensors_available():
        return get_package_version('safetensors')
    else:
        raise PackageNotFoundError('safetensors')

def device():
    """
    Get the device for computation (CPU or CUDA).

    Returns:
        str: A string indicating the device being used.
    """
    if is_torch_cuda_available():
        return 'cuda'
    else:
        return 'cpu'

if __name__ == '__main__':
    print(get_package_version('torch'))
