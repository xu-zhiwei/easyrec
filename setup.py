from setuptools import find_packages, setup

setup(
    name='pyrec',
    version='0.0.1',
    description='Easy-to-use implementations of well-known recommender system algorithms based on Python Tensorflow 2.',
    long_description='',
    keywords='Recommender System; CTR Estimation',
    license='MIT License',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
    url='https://github.com/xu-zhiwei/easyrec',
    author='Zhiwei Xu',
    author_email='zhiweixuchn@gmail.com'
)
