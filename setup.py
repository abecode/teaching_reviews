from setuptools import setup, find_packages

setup(
    name='teaching_reviews',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        # list your dependencies here
    ],
    url='https://github.com/abecode/teaching_reviews',
    author='Abe Kazemzadeh',
    author_email='abe.kazemzadeh@stthomas.edu',
    description='trying to understand teaching reviews with AI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='CC Share-Alike With-Attribution',
)
