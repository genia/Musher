import os
import sys
import platform
import subprocess
import shutil
import glob
import distutils.cmd
import codecs
import re

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils import sysconfig

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

readme_note = """\
.. note::

   For the latest source, discussion, etc, please visit the
   `GitHub repository <https://github.com/jmaldon1/Musher>`_\n\n

"""

with codecs.open('README.md', encoding='utf-8') as fobj:
    long_description = readme_note + fobj.read()

# List of all C++ test names
cpp_tests_list = [
    'test_musher_library',
    'test_musher_utils',
    'test_peak_detection',
]


class BuildCppTests(distutils.cmd.Command):
    description = 'Build c++ tests for musher library'
    user_options = [
        ('debug', None, 'sets config to Debug, config set to Release by default'),
        ('run-tests', 'r', 'set to run tests after building'),
        # https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#running-a-subset-of-the-tests
        ('filter=', 'f', 'Choose which tests to run, this will be passed to --gtest_filter'),
    ]

    def initialize_options(self):
        self.debug = False
        self.run_tests = False
        self.filter = ""

    def finalize_options(self):
        pass

    def run(self):
        # extdir = os.path.abspath(
        #     os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ROOT_DIR]

        if sysconfig.get_config_var('CXX') == "g++":  # Check if using g++ to compile
            # Check if user has g++ 8, if they do, use it to compile
            if os.path.exists("/usr/bin/g++-8"):
                cmake_args += ["-DCMAKE_CXX_COMPILER=/usr/bin/g++-8"]

        # Do not compile python module when building c++ code.
        cmake_args += ['-DBUILD_PYTHON_MODULE=OFF']

        if self.debug:
            cmake_args += ['-DCMAKE_BUILD_TYPE=Debug']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                ROOT_DIR)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        build_dir = 'build/'
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        subprocess.check_call(['cmake', ROOT_DIR] + cmake_args,
                              cwd=build_dir, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=build_dir)
        print()  # Add an empty line for cleaner output

        if self.run_tests:
            print("running tests...")
            test_bin_dir = os.path.abspath("./test_bin")
            for test in cpp_tests_list:
                if platform.system() == "Windows":
                    test_file_path = os.path.join(cfg, f"{test}.exe")
                    test_path = os.path.abspath(os.path.join(test_bin_dir, test_file_path))
                else:
                    test_file_path = test
                    test_path = os.path.abspath(os.path.join(test_bin_dir, test_file_path))
                print("=" * 35)
                print(f"Running Test '{test}'")
                print("=" * 35)
                test_args = [test_path]
                if self.filter:
                    test_args += [f"--gtest_filter={self.filter}"]
                    subprocess.check_call(test_args, cwd=ROOT_DIR)
                else:
                    subprocess.check_call(test_args, cwd=ROOT_DIR)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        # Output library to root directory
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        if sysconfig.get_config_var('CXX') == "g++":  # Check if using g++ to compile
            # Check if user has g++ 8, if they do, use it to compile
            if os.path.exists("/usr/bin/g++-8"):
                cmake_args += ["-DCMAKE_CXX_COMPILER=/usr/bin/g++-8"]

        # Do not build c++ tests when packaging code.
        cmake_args += ['-DBUILD_TESTING=OFF']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            # Output windows library to root directory
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


class CleanBuildCommand(distutils.cmd.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cleanup_list = [
            # Folders
            os.path.join(ROOT_DIR, "build"),
            os.path.join(ROOT_DIR, "dist"),
            os.path.join(ROOT_DIR, "musher.egg-info"),
            os.path.join(ROOT_DIR, ".eggs"),
            os.path.join(ROOT_DIR, ".pytest_cache"),
            os.path.join(ROOT_DIR, ".tox"),
            os.path.join(ROOT_DIR, "Release"),
            os.path.join(ROOT_DIR, "Debug"),
            os.path.join(ROOT_DIR, "test_bin"),
            # Files
            *glob.glob(os.path.join(ROOT_DIR, "*.so")),  # clean up linux outputs
            *glob.glob(os.path.join(ROOT_DIR, "*.dylib")),
            *glob.glob(os.path.join(ROOT_DIR, "*.pyd")),  # clean up windows outputs
            *glob.glob(os.path.join(ROOT_DIR, "*.dll")),
            *glob.glob(os.path.join(ROOT_DIR, "*.exe")),
            *glob.glob(os.path.join(ROOT_DIR, "*.exp")),
            *glob.glob(os.path.join(ROOT_DIR, "*.lib")),
        ]

        for item in cleanup_list:
            try:  # If item is a dir then remove it
                shutil.rmtree(item)
                print(f"cleaned {item}")
            except OSError:
                try:  # If item is a file then remove it
                    os.remove(item)
                    print(f"cleaned {item}")
                except OSError:
                    pass

        print(u'\u2713', "cleaning done")


setup(
    name='musher',
    version='0.1',
    description='Mush songs together',
    packages=find_packages(),
    # packages=['musher'],
    # add extension module
    ext_modules=[CMakeExtension("musher")],
    # add custom build_ext command
    cmdclass={"build_ext": CMakeBuild,
              "clean": CleanBuildCommand,
              "build_cpp_tests": BuildCppTests,
              },
    zip_safe=False,
    long_description=long_description,
    author='Brian Silver, Joshua Maldonado',
    author_email='joshjm9915@gmail.com',
    url='https://github.com/jmaldon1/Musher',
    license='MIT',
)
