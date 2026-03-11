// Copyright 2025 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/gcs/store_client/lmdb_store_client.h"

namespace ray::gcs {

LMDBStoreClient::LMDBStoreClient(instrumented_io_context &io_context,
                                 std::string lmdb_path)
    : io_context_(io_context), lmdb_path_(std::move(lmdb_path)) {
  RAY_LOG(INFO) << "Initializing LMDBStoreClient at " << lmdb_path_;
  // TODO: Open LMDB environment and databases for each table.
}

void LMDBStoreClient::AsyncPut(const std::string &table_name,
                               const std::string &key,
                               std::string data,
                               bool overwrite,
                               Postable<void(bool)> callback) {
  // TODO: Implement LMDB Put.
  // Wrap synchronous LMDB call in io_context.post to keep it async.
  io_context_.post(
      [callback = std::move(callback)]() mutable {
        std::move(callback).Post("LMDB.Put", true);
      },
      "LMDB.Put");
}

void LMDBStoreClient::AsyncGet(
    const std::string &table_name,
    const std::string &key,
    ToPostable<rpc::OptionalItemCallback<std::string>> callback) {
  // TODO: Implement LMDB Get.
}

void LMDBStoreClient::AsyncGetAll(
    const std::string &table_name,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  // TODO: Implement LMDB GetAll.
}

void LMDBStoreClient::AsyncMultiGet(
    const std::string &table_name,
    const std::vector<std::string> &keys,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  // TODO: Implement LMDB MultiGet.
}

void LMDBStoreClient::AsyncDelete(const std::string &table_name,
                                  const std::string &key,
                                  Postable<void(bool)> callback) {
  // TODO: Implement LMDB Delete.
}

void LMDBStoreClient::AsyncBatchDelete(const std::string &table_name,
                                       const std::vector<std::string> &keys,
                                       Postable<void(int64_t)> callback) {
  // TODO: Implement LMDB BatchDelete.
}

void LMDBStoreClient::AsyncGetNextJobID(Postable<void(int)> callback) {
  // TODO: Implement LMDB JobID increment.
}

void LMDBStoreClient::AsyncGetKeys(const std::string &table_name,
                                   const std::string &prefix,
                                   Postable<void(std::vector<std::string>)> callback) {
  // TODO: Implement LMDB prefix scan.
}

void LMDBStoreClient::AsyncExists(const std::string &table_name,
                                  const std::string &key,
                                  Postable<void(bool)> callback) {
  // TODO: Implement LMDB Exists.
}

}  // namespace ray::gcs
