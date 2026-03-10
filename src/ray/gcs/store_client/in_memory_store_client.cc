// Copyright 2017 The Ray Authors.
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

#include "ray/gcs/store_client/in_memory_store_client.h"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "ray/util/filesystem.h"
#include "src/ray/protobuf/gcs_snapshot.pb.h"

namespace ray::gcs {

InMemoryStoreClient::InMemoryStoreClient(instrumented_io_context &io_context,
                                         std::string snapshot_path,
                                         uint64_t snapshot_period_ms)
    : snapshot_path_(std::move(snapshot_path)) {
  if (snapshot_period_ms > 0 && !snapshot_path_.empty()) {
    periodical_runner_ = PeriodicalRunner::Create(io_context);
    periodical_runner_->RunFnPeriodically(
        [this] { DoSnapshot(); }, snapshot_period_ms, "InMemoryStoreClient.DoSnapshot");
  }
}

void InMemoryStoreClient::AsyncPut(const std::string &table_name,
                                   const std::string &key,
                                   std::string data,
                                   bool overwrite,
                                   Postable<void(bool)> callback) {
  auto &table = GetOrCreateMutableTable(table_name);
  bool inserted = false;
  if (overwrite) {
    inserted = table.InsertOrAssign(key, std::move(data));
  } else {
    inserted = table.Emplace(key, std::move(data));
  }
  std::move(callback).Post("GcsInMemoryStore.Put", inserted);
}

void InMemoryStoreClient::AsyncGet(
    const std::string &table_name,
    const std::string &key,
    ToPostable<rpc::OptionalItemCallback<std::string>> callback) {
  auto table = GetTable(table_name);
  std::optional<std::string> data;
  if (table != nullptr) {
    data = table->Get(key);
  }
  std::move(callback).Post("GcsInMemoryStore.Get", Status::OK(), std::move(data));
}

void InMemoryStoreClient::AsyncGetAll(
    const std::string &table_name,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  auto result = absl::flat_hash_map<std::string, std::string>();
  auto table = GetTable(table_name);
  if (table != nullptr) {
    result = table->GetMapClone();
  }
  std::move(callback).Post("GcsInMemoryStore.GetAll", std::move(result));
}

void InMemoryStoreClient::AsyncMultiGet(
    const std::string &table_name,
    const std::vector<std::string> &keys,
    Postable<void(absl::flat_hash_map<std::string, std::string>)> callback) {
  auto result = absl::flat_hash_map<std::string, std::string>();
  auto table = GetTable(table_name);
  if (table != nullptr) {
    table->ReadVisit(absl::MakeSpan(keys),
                     [&](std::string_view key, std::string_view value) {
                       result.emplace(key, value);
                     });
  }
  std::move(callback).Post("GcsInMemoryStore.GetAll", std::move(result));
}

void InMemoryStoreClient::AsyncDelete(const std::string &table_name,
                                      const std::string &key,
                                      Postable<void(bool)> callback) {
  auto &table = GetOrCreateMutableTable(table_name);
  auto erased = table.Erase(key);
  std::move(callback).Post("GcsInMemoryStore.Delete", erased);
}

void InMemoryStoreClient::AsyncBatchDelete(const std::string &table_name,
                                           const std::vector<std::string> &keys,
                                           Postable<void(int64_t)> callback) {
  auto &table = GetOrCreateMutableTable(table_name);
  int64_t num_erased = table.EraseKeys(absl::MakeSpan(keys));
  std::move(callback).Post("GcsInMemoryStore.BatchDelete", num_erased);
}

void InMemoryStoreClient::AsyncGetNextJobID(Postable<void(int)> callback) {
  auto job_id = job_id_.fetch_add(1, std::memory_order_acq_rel);
  std::move(callback).Post("GcsInMemoryStore.GetNextJobID", job_id);
}

ConcurrentFlatMap<std::string, std::string> &InMemoryStoreClient::GetOrCreateMutableTable(
    const std::string &table_name) {
  absl::WriterMutexLock lock(&mutex_);
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    return iter->second;
  }
  return tables_[table_name];
}

const ConcurrentFlatMap<std::string, std::string> *InMemoryStoreClient::GetTable(
    const std::string &table_name) {
  absl::ReaderMutexLock lock(&mutex_);
  auto iter = tables_.find(table_name);
  if (iter != tables_.end()) {
    return &iter->second;
  }
  return nullptr;
}

void InMemoryStoreClient::AsyncGetKeys(
    const std::string &table_name,
    const std::string &prefix,
    Postable<void(std::vector<std::string>)> callback) {
  std::vector<std::string> result;
  auto table = GetTable(table_name);
  if (table != nullptr) {
    table->ReadVisitAll([&](const std::string &key, const std::string &value) {
      if (absl::StartsWith(key, prefix)) {
        result.push_back(key);
      }
    });
  }
  std::move(callback).Post("GcsInMemoryStore.Keys", std::move(result));
}

void InMemoryStoreClient::AsyncExists(const std::string &table_name,
                                      const std::string &key,
                                      Postable<void(bool)> callback) {
  bool result = false;
  auto table = GetTable(table_name);
  if (table != nullptr) {
    result = table->Contains(key);
  }
  std::move(callback).Post("GcsInMemoryStore.Exists", result);
}

void InMemoryStoreClient::TriggerSnapshot() { DoSnapshot(); }

void InMemoryStoreClient::DoSnapshot() {
  if (snapshot_path_.empty()) {
    return;
  }
  RAY_LOG(DEBUG) << "Doing GCS snapshot to " << snapshot_path_;
  rpc::GcsSnapshot snapshot;
  {
    absl::ReaderMutexLock lock(&mutex_);
    for (const auto &it : tables_) {
      auto *table_proto = snapshot.add_tables();
      table_proto->set_table_name(it.first);
      auto map = it.second.GetMapClone();
      for (const auto &entry : map) {
        (*table_proto->mutable_entries())[entry.first] = entry.second;
      }
    }
    snapshot.set_job_id(job_id_.load());
  }

  std::string serialized;
  if (snapshot.SerializeToString(&serialized)) {
    std::ofstream os(snapshot_path_, std::ios::binary | std::ios::trunc);
    os << serialized;
    os.close();
    RAY_LOG(DEBUG) << "GCS snapshot done.";
  } else {
    RAY_LOG(ERROR) << "Failed to serialize GCS snapshot.";
  }
}

Status InMemoryStoreClient::LoadSnapshot() {
  if (snapshot_path_.empty()) {
    return Status::OK();
  }

  std::ifstream is(snapshot_path_, std::ios::binary);
  if (!is.is_open()) {
    RAY_LOG(INFO) << "No GCS snapshot found at " << snapshot_path_;
    return Status::OK();
  }

  RAY_LOG(INFO) << "Loading GCS snapshot from " << snapshot_path_;
  std::string serialized((std::istreambuf_iterator<char>(is)),
                         std::istreambuf_iterator<char>());
  is.close();

  rpc::GcsSnapshot snapshot;
  if (!snapshot.ParseFromString(serialized)) {
    return Status::Invalid("Failed to parse GCS snapshot.");
  }

  {
    absl::WriterMutexLock lock(&mutex_);
    for (const auto &table_proto : snapshot.tables()) {
      auto &table = tables_[table_proto.table_name()];
      for (const auto &entry : table_proto.entries()) {
        table.InsertOrAssign(entry.first, entry.second);
      }
    }
    job_id_ = snapshot.job_id();
  }

  RAY_LOG(INFO) << "GCS snapshot loaded successfully.";
  return Status::OK();
}

}  // namespace ray::gcs
