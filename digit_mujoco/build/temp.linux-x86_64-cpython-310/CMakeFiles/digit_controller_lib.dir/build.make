# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wzhu328/sac_ae_lidar_map/digit_mujoco

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310

# Include any dependencies generated for this target.
include CMakeFiles/digit_controller_lib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/digit_controller_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/digit_controller_lib.dir/flags.make

CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.o: CMakeFiles/digit_controller_lib.dir/flags.make
CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.o: ../../DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.o -c /home/wzhu328/sac_ae_lidar_map/digit_mujoco/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp

CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wzhu328/sac_ae_lidar_map/digit_mujoco/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp > CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.i

CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wzhu328/sac_ae_lidar_map/digit_mujoco/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp -o CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.s

# Object files for target digit_controller_lib
digit_controller_lib_OBJECTS = \
"CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.o"

# External object files for target digit_controller_lib
digit_controller_lib_EXTERNAL_OBJECTS =

../../DigitControlPybind/bin/libdigit_controller_lib.a: CMakeFiles/digit_controller_lib.dir/DigitControlPybind/src/include/digit_controller/src/digit_controller.cpp.o
../../DigitControlPybind/bin/libdigit_controller_lib.a: CMakeFiles/digit_controller_lib.dir/build.make
../../DigitControlPybind/bin/libdigit_controller_lib.a: CMakeFiles/digit_controller_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../DigitControlPybind/bin/libdigit_controller_lib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/digit_controller_lib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/digit_controller_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/digit_controller_lib.dir/build: ../../DigitControlPybind/bin/libdigit_controller_lib.a

.PHONY : CMakeFiles/digit_controller_lib.dir/build

CMakeFiles/digit_controller_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/digit_controller_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/digit_controller_lib.dir/clean

CMakeFiles/digit_controller_lib.dir/depend:
	cd /home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wzhu328/sac_ae_lidar_map/digit_mujoco /home/wzhu328/sac_ae_lidar_map/digit_mujoco /home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310 /home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310 /home/wzhu328/sac_ae_lidar_map/digit_mujoco/build/temp.linux-x86_64-cpython-310/CMakeFiles/digit_controller_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/digit_controller_lib.dir/depend

