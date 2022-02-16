from setuptools import find_packages, setup
import os

version_file = "fastvqa/version.py"


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def get_requirements(filename="requirements.txt"):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


setup(
    name="fastvqa",
    version=get_version(),
    description="Very Efficient End-to-End VQA Toolbox",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Timothy H. Wu",
    author_email="realtimothyhwu@gmail.com",
    keywords="computer vision, video quality assessment",
    url="https://github.com/timothyhtimothy/fast-vqa",
    include_package_data=True,
    packages=find_packages(exclude=("demos", "examplar_data_labels", "results")),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    license="Apache License 2.0",
    setup_requires=["numpy"],
    install_requires=get_requirements(),
    ext_modules=[],
    cmdclass={},
    zip_safe=False,
)
