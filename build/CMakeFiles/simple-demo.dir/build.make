# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /mnt/tera/code/cpp/projects/starfinder

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/tera/code/cpp/projects/starfinder/build

# Include any dependencies generated for this target.
include CMakeFiles/simple-demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/simple-demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/simple-demo.dir/flags.make

CMakeFiles/simple-demo.dir/main.cpp.o: CMakeFiles/simple-demo.dir/flags.make
CMakeFiles/simple-demo.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/tera/code/cpp/projects/starfinder/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/simple-demo.dir/main.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple-demo.dir/main.cpp.o -c /mnt/tera/code/cpp/projects/starfinder/main.cpp

CMakeFiles/simple-demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple-demo.dir/main.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/tera/code/cpp/projects/starfinder/main.cpp > CMakeFiles/simple-demo.dir/main.cpp.i

CMakeFiles/simple-demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple-demo.dir/main.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/tera/code/cpp/projects/starfinder/main.cpp -o CMakeFiles/simple-demo.dir/main.cpp.s

CMakeFiles/simple-demo.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/simple-demo.dir/main.cpp.o.requires

CMakeFiles/simple-demo.dir/main.cpp.o.provides: CMakeFiles/simple-demo.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/simple-demo.dir/build.make CMakeFiles/simple-demo.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/simple-demo.dir/main.cpp.o.provides

CMakeFiles/simple-demo.dir/main.cpp.o.provides.build: CMakeFiles/simple-demo.dir/main.cpp.o


# Object files for target simple-demo
simple__demo_OBJECTS = \
"CMakeFiles/simple-demo.dir/main.cpp.o"

# External object files for target simple-demo
simple__demo_EXTERNAL_OBJECTS =

simple-demo: CMakeFiles/simple-demo.dir/main.cpp.o
simple-demo: CMakeFiles/simple-demo.dir/build.make
simple-demo: /usr/local/lib/libopencv_gapi.so.4.2.0
simple-demo: /usr/local/lib/libopencv_stitching.so.4.2.0
simple-demo: /usr/local/lib/libopencv_aruco.so.4.2.0
simple-demo: /usr/local/lib/libopencv_bgsegm.so.4.2.0
simple-demo: /usr/local/lib/libopencv_bioinspired.so.4.2.0
simple-demo: /usr/local/lib/libopencv_ccalib.so.4.2.0
simple-demo: /usr/local/lib/libopencv_dnn_objdetect.so.4.2.0
simple-demo: /usr/local/lib/libopencv_dnn_superres.so.4.2.0
simple-demo: /usr/local/lib/libopencv_dpm.so.4.2.0
simple-demo: /usr/local/lib/libopencv_face.so.4.2.0
simple-demo: /usr/local/lib/libopencv_freetype.so.4.2.0
simple-demo: /usr/local/lib/libopencv_fuzzy.so.4.2.0
simple-demo: /usr/local/lib/libopencv_hdf.so.4.2.0
simple-demo: /usr/local/lib/libopencv_hfs.so.4.2.0
simple-demo: /usr/local/lib/libopencv_img_hash.so.4.2.0
simple-demo: /usr/local/lib/libopencv_line_descriptor.so.4.2.0
simple-demo: /usr/local/lib/libopencv_quality.so.4.2.0
simple-demo: /usr/local/lib/libopencv_reg.so.4.2.0
simple-demo: /usr/local/lib/libopencv_rgbd.so.4.2.0
simple-demo: /usr/local/lib/libopencv_saliency.so.4.2.0
simple-demo: /usr/local/lib/libopencv_sfm.so.4.2.0
simple-demo: /usr/local/lib/libopencv_stereo.so.4.2.0
simple-demo: /usr/local/lib/libopencv_structured_light.so.4.2.0
simple-demo: /usr/local/lib/libopencv_superres.so.4.2.0
simple-demo: /usr/local/lib/libopencv_surface_matching.so.4.2.0
simple-demo: /usr/local/lib/libopencv_tracking.so.4.2.0
simple-demo: /usr/local/lib/libopencv_videostab.so.4.2.0
simple-demo: /usr/local/lib/libopencv_xfeatures2d.so.4.2.0
simple-demo: /usr/local/lib/libopencv_xobjdetect.so.4.2.0
simple-demo: /usr/local/lib/libopencv_xphoto.so.4.2.0
simple-demo: /usr/local/lib/libopencv_highgui.so.4.2.0
simple-demo: /usr/local/lib/libopencv_shape.so.4.2.0
simple-demo: /usr/local/lib/libopencv_datasets.so.4.2.0
simple-demo: /usr/local/lib/libopencv_plot.so.4.2.0
simple-demo: /usr/local/lib/libopencv_text.so.4.2.0
simple-demo: /usr/local/lib/libopencv_dnn.so.4.2.0
simple-demo: /usr/local/lib/libopencv_ml.so.4.2.0
simple-demo: /usr/local/lib/libopencv_phase_unwrapping.so.4.2.0
simple-demo: /usr/local/lib/libopencv_optflow.so.4.2.0
simple-demo: /usr/local/lib/libopencv_ximgproc.so.4.2.0
simple-demo: /usr/local/lib/libopencv_video.so.4.2.0
simple-demo: /usr/local/lib/libopencv_videoio.so.4.2.0
simple-demo: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
simple-demo: /usr/local/lib/libopencv_objdetect.so.4.2.0
simple-demo: /usr/local/lib/libopencv_calib3d.so.4.2.0
simple-demo: /usr/local/lib/libopencv_features2d.so.4.2.0
simple-demo: /usr/local/lib/libopencv_flann.so.4.2.0
simple-demo: /usr/local/lib/libopencv_photo.so.4.2.0
simple-demo: /usr/local/lib/libopencv_imgproc.so.4.2.0
simple-demo: /usr/local/lib/libopencv_core.so.4.2.0
simple-demo: CMakeFiles/simple-demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/tera/code/cpp/projects/starfinder/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable simple-demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple-demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/simple-demo.dir/build: simple-demo

.PHONY : CMakeFiles/simple-demo.dir/build

CMakeFiles/simple-demo.dir/requires: CMakeFiles/simple-demo.dir/main.cpp.o.requires

.PHONY : CMakeFiles/simple-demo.dir/requires

CMakeFiles/simple-demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/simple-demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/simple-demo.dir/clean

CMakeFiles/simple-demo.dir/depend:
	cd /mnt/tera/code/cpp/projects/starfinder/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/tera/code/cpp/projects/starfinder /mnt/tera/code/cpp/projects/starfinder /mnt/tera/code/cpp/projects/starfinder/build /mnt/tera/code/cpp/projects/starfinder/build /mnt/tera/code/cpp/projects/starfinder/build/CMakeFiles/simple-demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/simple-demo.dir/depend

