load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "lmdb",
    srcs = ["libraries/liblmdb/mdb.c", "libraries/liblmdb/midl.c"],
    hdrs = ["libraries/liblmdb/lmdb.h", "libraries/liblmdb/midl.h"],
    copts = [
        "-w",
    ],
    includes = ["libraries/liblmdb"],
    visibility = ["//visibility:public"],
)
