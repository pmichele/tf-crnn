from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-gpu>=1.8']

setup(
    name='tf_crnn',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={ 'tf_crnn': [ '*.json' ] },
    # data_files=[('./', 'tf_crnn/model_params.json')],
    include_package_data=True,
    description='hwr'
)
