from setuptools import find_packages, setup

setup(
    name="jatmo",
    version="1.0",
    description="Prompt injection defense by fine tuning",
    long_description="Prompt injection defense by fine tuning",
    packages=find_packages("src", exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.9",
    include_package_data=True,
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "jatmo-server-test=jatmo.test_server:main",
            "jatmo-prompt-select=jatmo.prompt_injection_select.main:main",
            "jatmo-autogen=jatmo.example_tasks.auto_tasks.main:main",
            "jatmo-semiauto=jatmo.example_tasks.semiauto_tasks.main:main",
        ]
    },
    zip_safe=False,
)
