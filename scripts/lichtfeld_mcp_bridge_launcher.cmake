cmake_minimum_required(VERSION 3.21)

if(WIN32)
    set(_python_names python py python3)
else()
    set(_python_names python3 python)
endif()

set(_version_probe "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)")
foreach(_python_name IN LISTS _python_names)
    unset(_python_candidate)
    find_program(_python_candidate NAMES "${_python_name}" NO_CACHE)
    if(NOT _python_candidate)
        continue()
    endif()

    execute_process(
        COMMAND "${_python_candidate}" -c "${_version_probe}"
        RESULT_VARIABLE _probe_result
        OUTPUT_QUIET
        ERROR_QUIET)
    if(_probe_result STREQUAL "0")
        set(_python_executable "${_python_candidate}")
        break()
    endif()
endforeach()

if(NOT _python_executable)
    message(FATAL_ERROR "LichtFeld MCP bridge requires Python 3.9 or newer")
endif()

if(NOT DEFINED LICHTFELD_MCP_BRIDGE_SCRIPT)
    set(LICHTFELD_MCP_BRIDGE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/lichtfeld_mcp_bridge.py")
endif()
if(NOT EXISTS "${LICHTFELD_MCP_BRIDGE_SCRIPT}")
    message(FATAL_ERROR "LichtFeld MCP bridge script not found: ${LICHTFELD_MCP_BRIDGE_SCRIPT}")
endif()

execute_process(
    COMMAND "${_python_executable}" "${LICHTFELD_MCP_BRIDGE_SCRIPT}"
    RESULT_VARIABLE _bridge_result)
if(NOT _bridge_result STREQUAL "0")
    message(FATAL_ERROR "LichtFeld MCP bridge exited with status ${_bridge_result}")
endif()
