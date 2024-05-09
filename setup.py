from setuptools import setup, find_packages


setup(
    name='montara',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    entry_points={
        'console_scripts': [
            'des-montara-make-input-cosmos-cat = montara:make_input_cosmos_cat_cli',
        ]
    }
)
