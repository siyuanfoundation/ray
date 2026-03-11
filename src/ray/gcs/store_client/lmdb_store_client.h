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

#pragma once

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "lmdb.h"
#include "ray/common/asio/instrumented_io_context.h"
#include "ray/gcs/store_client/store_client.h"

namespace ray::gcs {

/// \class LMDBStoreClient
/// Please refer to StoreClient for API semantics.
///
/// This class is thread safe.
class LMDBStoreClient : public StoreClient {
 public:
  explicit LMDBStoreClient(instrumented_io_context &io_context, std::string lmdb_path);

  ~LMDBStoreClient() override;

  void AsyncPut(const std::string &table_name,
                const std::string &key,
                std::string data,
                bool overwrite,
                Postable<void(bool)> callback) override;

  void AsyncGet(const std::string &table_name,
                const std::string &key,
                ToPostable<rpc::OptionalItemCallback<std::string>> callback) override;

  void AsyncGetAll(
      const std::string &table_name,
      Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) override;

  void AsyncMultiGet(
      const std::string &table_name,
      const std::vector<std::string> &keys,
      Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) override;

  void AsyncDelete(const std::string &table_name,
                   const std::string &key,
                   Postable<void(bool)> callback) override;

  void AsyncBatchDelete(const std::string &table_name,
                        const std::vector<std::string> &keys,
                        Postable<void(int64_t)> callback) override;

  void AsyncGetNextJobID(Postable<void(int)> callback) override;

  void AsyncGetKeys(const std::string &table_name,
                    const std::string &prefix,
                    Postable<void(std::vector<std::string>)> callback) override;

  void AsyncExists(const std::string &table_name,
                   const std::string &key,
                   Postable<void(bool)> callback) override;

 private:
  MDB_dbi GetOrCreateTable(MDB_txn *txn, const std::string &table_name);

  instrumented_io_context &io_context_;
  std::string lmdb_path_;

  MDB_env *env_ = nullptr;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, MDB_dbi> dbis_ ABSL_GUARDED_BY(mutex_);

  /// Dedicated thread for LMDB operations to avoid MDB_NOTLS issues.
  std::unique_ptr<std::thread> worker_thread_;
  instrumented_io_context worker_io_context_;
};

}  // namespace ray::gcs
