# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_detection_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED detection_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(detection_FOUND FALSE)
  elseif(NOT detection_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(detection_FOUND FALSE)
  endif()
  return()
endif()
set(_detection_CONFIG_INCLUDED TRUE)

# output package information
if(NOT detection_FIND_QUIETLY)
  message(STATUS "Found detection: 0.0.0 (${detection_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'detection' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${detection_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(detection_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${detection_DIR}/${_extra}")
endforeach()
