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

#include "ray/gcs/store_client/etcd_store_client.h"

namespace ray::gcs {

EtcdStoreClient::EtcdStoreClient(instrumented_io_context &io_context,
                                 std::vector<std::string> etcd_endpoints)
    : io_context_(io_context), etcd_endpoints_(std::move(etcd_endpoints)) {
  RAY_LOG(INFO) << "Initializing EtcdStoreClient with " << etcd_endpoints_.size()
                << " endpoints.";
  // TODO: Initialize etcd-cpp-apiv3 client.
}

void EtcdStoreClient::AsyncPut(const std::string &table_name,
                               const std::string &key,
                               std::string data,
                               bool overwrite,
                               Postable<void(bool)> callback) {
  // TODO: Implement etcd Put.
}

void EtcdStoreClient::AsyncGet(
    const std::string &table_name,
    const std::string &key,
    ToPostable<rpc::OptionalItemCallback<std::string>> callback) {
  // TODO: Implement etcd Get.
}

void EtcdStoreClient::AsyncGetAll(
    const std::string &table_name,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  // TODO: Implement etcd GetAll (prefix scan).
}

void EtcdStoreClient::AsyncMultiGet(
    const std::string &table_name,
    const std::vector<std::string> &keys,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  // TODO: Implement etcd MultiGet (transaction).
}

void EtcdStoreClient::AsyncDelete(const std::string &table_name,
                                  const std::string &key,
                                  Postable<void(bool)> callback) {
  // TODO: Implement etcd Delete.
}

void EtcdStoreClient::AsyncBatchDelete(const std::string &table_name,
                                       const std::vector<std::string> &keys,
                                       Postable<void(int64_t)> callback) {
  // TODO: Implement etcd BatchDelete.
}

void EtcdStoreClient::AsyncGetNextJobID(Postable<void(int)> callback) {
  // TODO: Implement etcd atomic increment.
}

void EtcdStoreClient::AsyncGetKeys(const std::string &table_name,
                                   const std::string &prefix,
                                   Postable<void(std::vector<std::string>)> callback) {
  // TODO: Implement etcd prefix scan for keys.
}

void EtcdStoreClient::AsyncExists(const std::string &table_name,
                                  const std::string &key,
                                  Postable<void(bool)> callback) {
  // TODO: Implement etcd Exists.
}

}  // namespace ray::gcs
