from setuptools import setup, find_packages

def get_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name='wave-mamba-map',
    version='0.1.0',
    description='Wavelet-enhanced image project',
    packages=find_packages(exclude=('ckpt', 'test_results', 'options')),
    install_requires=get_requirements(),
)
