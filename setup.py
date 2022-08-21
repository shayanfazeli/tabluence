from setuptools import setup, find_packages

setup(
    name="tabluence",
    version="0.0.1",
    description="Tabluence: Tabular Time-series Machine Learning Framework",
    url="https://github.com/shayanfazeli/tabluence",
    author="Shayan Fazeli",
    author_email="shayan.fazeli@gmail.com",
    license="Apache",
    classifiers=[
        'Intended Audience :: Science/Research',
        #'Development Status :: 1 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords="machine learning,coronavirus,deep learning,inference",
    packages=find_packages(),
    python_requires='>3.6.0',
    scripts=[
        'tabluence/bin/tabluence_train',
    ],
    install_requires=[],
    zip_safe=False
)