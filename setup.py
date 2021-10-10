import setuptools

setuptools.setup(
    name="cgb",
    version="0.0.1",
    author="Seyedsaman Emami, Gonzalo Martínez-Muñoz",
    author_email="emami.seyedsaman@uam.es, gonzalo.martinez@uam.es",
    description="Condensed Gradient Boosting Decision Tree",
    packages=['cgb'],
    install_requires=['numpy', 'scikit-learn', 'scipy', 'numbers'],
    classifiers=("Programming Language :: Python :: 3")
)