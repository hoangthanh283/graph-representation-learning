import pytest
from package_name import _module_from_package, _packages


@pytest.mark.parametrize("package", _packages)
def test_import_package_name_class(package):
    module = _module_from_package(package)
    exec(f'from package_name.{package}.model import {module}')
