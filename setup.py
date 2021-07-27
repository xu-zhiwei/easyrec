from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='easyrec-python',
    version='0.0.1',
    description='Easy-to-use implementations of well-known recommender system algorithms based on Python Tensorflow 2.',
    long_description=long_description,
    keywords='Recommender System; CTR Estimation',
    license='MIT License',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Utilities',
    ],
    url='https://github.com/xu-zhiwei/easyrec',
    author='Zhiwei Xu',
    author_email='zhiweixuchn@gmail.com'
)
