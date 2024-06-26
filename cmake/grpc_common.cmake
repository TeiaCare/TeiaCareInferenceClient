file(MAKE_DIRECTORY ${PROTO_OUT_DIR})
set(PROTO_INCLUDE "${PROTO_OUT_DIR}/${PROTO_FILENAME}.pb.h")
set(GRPC_INCLUDE "${PROTO_OUT_DIR}/${PROTO_FILENAME}.grpc.pb.h")
set(PROTO_SRC "${PROTO_OUT_DIR}/${PROTO_FILENAME}.pb.cc")
set(GRPC_SRC "${PROTO_OUT_DIR}/${PROTO_FILENAME}.grpc.pb.cc")

list(APPEND GRPC_PROTO_FILES
    ${PROTO_INCLUDE}
    ${GRPC_INCLUDE}
    ${PROTO_SRC}
    ${GRPC_SRC}
)

add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_INCLUDE} ${GRPC_SRC} ${GRPC_INCLUDE}
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --grpc_out=generate_mock_code=true:${PROTO_OUT_DIR} --cpp_out=${PROTO_OUT_DIR} --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_PROGRAM} ${PROTO_FILE}
    WORKING_DIRECTORY ${PROTO_IN_DIR}
    DEPENDS ${PROTO_IN_DIR}/${PROTO_FILE}
    COMMENT "Generating gRPC files"
    VERBATIM
)