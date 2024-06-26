function(add_warnings TARGET)
    message(STATUS "Compiler warnings enabled")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET} PRIVATE -Wall -Wextra)
    elseif(MSVC)
        target_compile_options(${TARGET} PRIVATE /W4)
    endif()
endfunction()

function(add_warnings_as_errors TARGET)
    message(STATUS "Treat warnings as errors enabled")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${TARGET} PRIVATE -Werror)
    elseif(MSVC)
        target_compile_options(${TARGET} PRIVATE /WX)
    endif()
endfunction()
