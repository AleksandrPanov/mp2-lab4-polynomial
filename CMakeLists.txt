cmake_minimum_required(VERSION 3.5)
project(PhysicTetris LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT DEFINED SFML_DIR)
    message(FATAL_ERROR "You must set SFML_DIR variable")
endif()
set(CMAKE_PREFIX_PATH "${SFML_DIR}/lib/cmake/SFML")

#Files directories
set(SRC "src/")
set(EXTERN "extern/")

#Files
set(RESOURES "${SRC}/PhysicTetris.rc" "${SRC}/resource.h")

FILE(GLOB UTILS "${SRC}/*.hpp")

FILE(GLOB PARTICLE_SYSTEM 
	"${SRC}/ParticleSystem/*.hpp"
	"${SRC}/ParticleSystem/Affector/*.hpp"
	"${SRC}/ParticleSystem/Emittor/*.hpp")

FILE(GLOB STATES "${SRC}/States/*.hpp")

FILE(GLOB TETRIS_LOGIC 
	"${SRC}/TetrisLogic/*.hpp"
	"${SRC}/TetrisLogic/Figure/*.hpp"
	"${SRC}/TetrisLogic/Rules/*.hpp")

FILE(GLOB TETRIS_VISUAL "${SRC}/TetrisVisual/*.hpp")

#executable
add_executable(${PROJECT_NAME} "${SRC}/main.cpp" ${UTILS} ${PARTICLE_SYSTEM} ${STATES} ${TETRIS_LOGIC} ${TETRIS_VISUAL} ${RESOURES})

#BOX2D
target_include_directories(${PROJECT_NAME} PUBLIC "extern/box2d/include")
if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
	target_link_directories(${PROJECT_NAME} PUBLIC "${EXTERN}box2d/lib/x64/Debug")
else ()
	target_link_directories(${PROJECT_NAME} PUBLIC "${EXTERN}box2d/lib/x64/Release")
endif (CMAKE_BUILD_TYPE EQUAL "DEBUG")

#SFML
find_package(SFML 2.5 COMPONENTS audio graphics window system REQUIRED)
target_link_directories(${PROJECT_NAME} PUBLIC "${SFML_DIR}/../..")

#Link libraries
if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    target_link_libraries(${PROJECT_NAME} sfml-system sfml-window sfml-graphics sfml-audio box2d)
else ()
	target_link_libraries(${PROJECT_NAME} sfml-system-s sfml-window-s sfml-graphics-s sfml-audio-s opengl32 freetype gdi32 winmm sfml-main openal32 flac vorbisenc vorbisfile vorbis ogg box2d)
endif (CMAKE_BUILD_TYPE EQUAL "DEBUG")