from setuptools import Extension, dist, find_packages, setup

setup(
    name='pyrec',
    version='0.0.1',
    description='Python tensorflow2 implementations of recommender system algorithms',
    long_description='',
    keywords='recommender system',
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
    url='https://github.com/xu-zhiwei/pyrec',
    author='Zhiwei Xu',
    author_email='zhiweixuchn@gmail.com'
    )